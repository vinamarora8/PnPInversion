from pathlib import Path
import random
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    T5Tokenizer,
    T5EncoderModel,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from diffusers import DiffusionPipeline, DDPMScheduler
from tqdm import tqdm
from PIL import Image
from laion_dataset import LaionFolderDataset
import wandb
from PIL import ImageDraw, ImageFont
import argparse
import shutil

from adapter import TextEncoderAdapter


# NEG_PROMPT = "blurry, low quality, ugly"
NEG_PROMPT = "ugly, blurry, black, low res, unrealistic"
RUN_ID = "laion_sd1p5_t5_v1_1_xxl_mlp"
# RUN_ID = "sanity"
OUTPUT_DIR = Path(__file__).parent / "outputs" / RUN_ID
TEST_EVERY_N_STEP = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite:
        shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_image_dir = OUTPUT_DIR / "test_images"
    test_image_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = OUTPUT_DIR / "adapter.pth"

    seed_everything(42)

    # dataset = AdapterDataset()
    # dataset = LaionStreamingDataset("laion/laion2B-en-aesthetic", split="train")
    dataset = LaionFolderDataset("laion2B-en-aesthetic/laion_100gb_512px/")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)


    sd_model_key = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    # t5_model_key = "google-t5/t5-large"
    t5_model_key = "google/t5-v1_1-xxl"
    device = "cuda"

    sd_pipe = DiffusionPipeline.from_pretrained(
        sd_model_key,
        # torch_dtype=torch.bfloat16,
        safety_checker=None,
    ).to(device)
    max_text_seq_length = sd_pipe.text_encoder.config.max_position_embeddings
    # sd_pipe.enable_xformers_memory_efficient_attention()
    sd_pipe.set_progress_bar_config(leave=False)

    vae = sd_pipe.vae
    unet = sd_pipe.unet
    scheduler = DDPMScheduler.from_pretrained(sd_model_key, subfolder="scheduler")

    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_key)
    t5_encoder = (
        T5EncoderModel.from_pretrained(t5_model_key).to(device)
        # .half()
        .requires_grad_(False)
    )
    t5_encoder.eval()

    adapter = (
        TextEncoderAdapter(
            input_dim=t5_encoder.config.d_model,
            output_dim=sd_pipe.text_encoder.config.hidden_size,
        ).to(device)
        # .half()
    )
    print(adapter)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    t5_encoder.requires_grad_(False)
    adapter.requires_grad_(True)

    optimizer = AdamW(adapter.parameters(), lr=1e-4)

    initial_step = 0
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path)
        adapter.load_state_dict(checkpoint["adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        initial_step = checkpoint["step"] + 1
        print(f"Loaded checkpoint from {ckpt_path} at step {initial_step}")

    print(
        f"Num trainable parameters: {sum(p.numel() for p in adapter.parameters() if p.requires_grad)}"
    )

    # sanity_batch = next(iter(dataloader))
    # noise = None
    # timesteps = None

    wandb.init(project="adapt-t5", entity="vinam-arora8", name=RUN_ID)

    step = initial_step
    while True:
        for clean_images, captions in (pbar := tqdm(dataloader, initial=initial_step)):
        # for _ in (pbar := tqdm(range(len(dataloader)), initial=initial_step)):
            # clean_images, captions = sanity_batch

            if step % TEST_EVERY_N_STEP == 0:
                try:
                    adapter.eval()
                    test_image = test_generation(
                        t5_tokenizer, t5_encoder, adapter, sd_pipe, captions[0]
                    )
                    save_path = test_image_dir / f"{step}.png"
                    save_image_with_caption(
                        image=test_image,
                        caption=captions[0],
                        save_path=save_path,
                    )
                except Exception as e:
                    print(f"Error saving test image: {e}")

                adapter.train()
                torch.save(
                    {
                        "adapter": adapter.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                    },
                    ckpt_path,
                )

            clean_images = clean_images.to(device)  # [B, 3, 512, 512]

            with torch.no_grad():
                latents = vae.encode(clean_images).latent_dist.sample()
                latents = latents * 0.18215  # Magic scaling number for SD 1.5

            # B. ADD NOISE (Standard SD stuff)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                text_emb, mask = encode_prompt(
                    prompts=captions,
                    tokenizer=t5_tokenizer,
                    text_encoder=t5_encoder,
                    max_length=max_text_seq_length,
                )
                text_emb = text_emb.float()

            adapted_embeds = adapter(text_emb, mask)

            # UNET
            noise_pred = unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=adapted_embeds,  # <--- INJECTED HERE
                # encoder_attention_mask=mask,
            ).sample

            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(
                f"Loss = {loss.item():.3f}, Mask = {mask.float().mean().item():.3f}"
                + f", Max = {clean_images.max().item():.3f}, Min = {clean_images.min().item():.3f}"
            )
            step += 1

            wandb.log(
                {
                    "loss": loss.item(),
                },
                step=step,
            )


@torch.inference_mode()
def test_generation(
    tokenizer,
    text_encoder,
    adapter,
    sd_pipe,
    prompt="Cafe Latte in Round Red Cup and Saucer",
):
    embeds, mask = encode_prompt(
        [prompt, NEG_PROMPT],
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        max_length=sd_pipe.text_encoder.config.max_position_embeddings,
    )
    # prompt_embeds, neg_embeds = embeds.chunk(2, dim=0)
    prompt_embeds, neg_embeds = adapter(embeds, mask).chunk(2, dim=0)

    image = sd_pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        num_inference_steps=50,
        guidance_scale=1.0,
        # attention_mask=mask,
    ).images[0]

    return image




def encode_prompt(
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    text_encoder: PreTrainedModel,
    max_length=77,
):
    if isinstance(prompts, str):
        prompts = [prompts]

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    ).to(text_encoder.device)

    with torch.no_grad():
        text_embeds = text_encoder(
            text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        ).last_hidden_state

    # mask = text_inputs.attention_mask.unsqueeze(-1).expand_as(text_embeds)
    # text_embeds = text_embeds * mask.to(text_embeds.device)

    return text_embeds, text_inputs.attention_mask


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def save_image_with_caption(image: Image, caption: str, save_path: Path):
    # Copy image to editable format
    img_with_caption = image.copy()
    draw = ImageDraw.Draw(img_with_caption)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    # Calculate text size
    try:
        # PIL >= 8.0.0
        bbox = font.getbbox(caption)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            text_width, text_height = font.getsize(caption)
        except Exception:
            # fallback
            text_width, text_height = 200, 30
    padding = 16
    # Draw a white rectangle at the top large enough for the caption
    rect_x0, rect_y0 = 0, 0
    rect_x1 = min(img_with_caption.width, text_width + 2 * padding)
    rect_y1 = text_height + 2 * padding
    draw.rectangle([(rect_x0, rect_y0), (rect_x1, rect_y1)], fill=(255, 255, 255))
    # Draw the text over the white rectangle, slightly inset
    draw.text(
        (rect_x0 + padding, rect_y0 + padding),
        caption,
        fill=(0, 0, 0),
        font=font,
    )
    img_with_caption.save(save_path)


if __name__ == "__main__":
    main()
