from pathlib import Path
import random
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from torchvision import transforms
from transformers import (
    T5Tokenizer,
    T5EncoderModel,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from diffusers import DiffusionPipeline, DDPMScheduler
from tqdm import tqdm


NEG_PROMPT = "blurry, low quality, ugly"
RUN_ID = "first"
OUTPUT_DIR = Path(__file__).parent / "outputs" / RUN_ID
TEST_EVERY_N_STEP = 50


def main():
    test_image_dir = OUTPUT_DIR / "test_images"
    test_image_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = OUTPUT_DIR / "adapter.pth"

    seed_everything(42)

    sd_model_key = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    t5_model_key = "google-t5/t5-base"
    device = "cuda"

    sd_pipe = DiffusionPipeline.from_pretrained(
        sd_model_key,
        # torch_dtype=torch.float16,
        # variant="fp16",
    ).to(device)
    max_text_seq_length = sd_pipe.text_encoder.config.max_position_embeddings
    sd_pipe.enable_xformers_memory_efficient_attention()
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

    adapter = (
        TextEncoderAdapter(
            input_dim=t5_encoder.config.d_model,
            output_dim=sd_pipe.text_encoder.config.hidden_size,
        ).to(device)
        # .half()
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    t5_encoder.requires_grad_(False)
    adapter.requires_grad_(True)

    initial_step = 0
    if ckpt_path.exists():
        adapter.load_state_dict(torch.load(ckpt_path))
        print(f"Loaded adapter from {ckpt_path}")

        test_image_files = list(test_image_dir.glob("*.png"))
        test_image_files.sort(key=lambda x: int(x.stem))
        last_test_image_file = test_image_files[-1]
        initial_step = int(last_test_image_file.stem) + 1
        print(f"Initial step: {initial_step}")

        

    optimizer = AdamW(adapter.parameters(), lr=1e-4)

    dataset = AdapterDataset()
    dataloader = DataLoader(dataset, batch_size=4, prefetch_factor=2, num_workers=1)

    step = initial_step
    with torch.autocast("cuda", torch.bfloat16):
        for clean_images, captions in (pbar := tqdm(dataloader, initial=initial_step)):
            if step % TEST_EVERY_N_STEP == 0:
                adapter.eval()
                test_image = test_generation(t5_tokenizer, t5_encoder, adapter, sd_pipe)
                save_path = test_image_dir / f"{step}.png"
                test_image.save(save_path)
                # print(f"Saved test image to {save_path}")
                adapter.train()

                torch.save(adapter.state_dict(), ckpt_path)

            clean_images = clean_images.to(device)  # [B, 3, 512, 512]
            batch_size = len(captions)

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
                text_emb = encode_prompt(
                    prompts=captions,
                    tokenizer=t5_tokenizer,
                    text_encoder=t5_encoder,
                    max_length=max_text_seq_length,
                )

            adapted_embeds = adapter(text_emb)

            # UNET
            noise_pred = unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=adapted_embeds,  # <--- INJECTED HERE
            ).sample

            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f"Loss = {loss.item():.3f}")
            step += 1
            # print(f"Batch {step} loaded! Shape: {clean_images.shape}")
            # print(f"Caption: {captions[0]}")
            # print(f"{adapted_embeds.shape}")

            # if step % 10 == 0:
            #     print(f"Step {step} Loss: {loss.item()}")


@torch.inference_mode()
@torch.autocast("cuda", torch.bfloat16)
def test_generation(tokenizer, text_encoder, adapter, sd_pipe):
    prompt = "a futuristic city with flying cars"
    embeds = encode_prompt(
        [prompt, NEG_PROMPT],
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        max_length=sd_pipe.text_encoder.config.max_position_embeddings,
    )
    # prompt_embeds, neg_embeds = embeds.chunk(2, dim=0)
    prompt_embeds, neg_embeds = adapter(embeds).chunk(2, dim=0)

    image = sd_pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images[0]

    return image


class TextEncoderAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, 4 * output_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(4 * output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.adapter(embeddings)


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
        text_embeds = text_encoder(text_inputs.input_ids).last_hidden_state

    return text_embeds


class AdapterDataset(IterableDataset):
    def __init__(self, split="train", buffer_size=1000):
        # 1. Use 'poloclub/diffusiondb' (contains real images, not just URLs)
        # '2m_first_1k' is a tiny subset good for verifying the loop starts.
        # For real training, use '2m_random_1k' or just stream the main '2m_first_10k' etc.
        # Alternatively: "lambdalabs/pokemon-blip-captions" is extremely fast.

        self.dataset = load_dataset(
            "poloclub/diffusiondb",
            # "2m_first_5k", # Load a specific small config for speed
            split=split,
            streaming=True,
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __iter__(self):
        for sample in self.dataset:
            try:
                # DiffusionDB keys: 'image' (PIL) and 'prompt' (Text)
                image = sample["image"]
                text = sample["prompt"]  # Note: key is 'prompt', not 'text'

                image_tensor = self.transform(image.convert("RGB"))
                yield image_tensor, text

            except Exception as e:
                print(f"Skipping bad sample: {e}")
                continue


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()
