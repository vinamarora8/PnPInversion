
import torch
from diffusers import DDIMScheduler
import numpy as np
from PIL import Image
import os
import json
import random
import argparse
import time

from utils.utils import txt_draw,load_512,latent2image

from pnp import PNP
from preprocess import Preprocess

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

NUM_DDIM_STEPS = 50

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start



# model_key = "runwayml/stable-diffusion-v1-5"
# model_key = "CompVis/stable-diffusion-v1-4"
model_key = "stable-diffusion-v1-5/stable-diffusion-v1-5"
toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
toy_scheduler.set_timesteps(NUM_DDIM_STEPS)

timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=NUM_DDIM_STEPS,
                                                           strength=1.0,
                                                           device=device)
model = Preprocess(device, model_key=model_key)
pnp = PNP(model_key)


def edit_image_ddim_PnP(
    image_path,
    prompt_src,
    prompt_tar,
    guidance_scale=7.5,
    image_shape=[512,512]
):
    #torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    _, rgb_reconstruction, latent_reconstruction = model.extract_latents(data_path=image_path,
                                         num_steps=NUM_DDIM_STEPS,
                                         inversion_prompt=prompt_src)
    
    edited_image=pnp.run_pnp(image_path,latent_reconstruction,prompt_tar,guidance_scale)
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(255*np.array(rgb_reconstruction[0].permute(1,2,0).cpu().detach())),
        np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
        ),1))



def edit_image_directinversion_PnP(
    image_path,
    prompt_src,
    prompt_tar,
    guidance_scale=7.5,
    image_shape=[512,512]
):
    #torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    inverted_x, rgb_reconstruction, _ = model.extract_latents(data_path=image_path,
                                         num_steps=NUM_DDIM_STEPS,
                                         inversion_prompt=prompt_src)

    edited_image=pnp.run_pnp(image_path,inverted_x,prompt_tar,guidance_scale)
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(np.array(latent2image(model=pnp.vae, latents=inverted_x[1].to(pnp.vae.dtype))[0])),
        np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
        ),1))


def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array

image_save_paths={
    "ddim+pnp":"ddim+pnp",
    "directinversion+pnp":"directinversion+pnp",
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    # parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["ddim+pnp","directinversion+pnp"]) # the editing methods that needed to run
    # parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["directinversion+pnp"]) # the editing methods that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["ddim+pnp"]) # the editing methods that needed to run
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list
    
    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)

    for key, item in editing_instruction.items():

        start_time = time.time()

        if item["editing_type_id"] not in edit_category_list:
            continue
        
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        editing_instruction = item["editing_instruction"]
        blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []
        mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")

        for edit_method in edit_method_list:
            present_image_save_path=image_path.replace(data_path, os.path.join(output_path,image_save_paths[edit_method]))
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"editing image [{image_path}] with [{edit_method}]")
                setup_seed()
                #torch.cuda.empty_cache()
                if edit_method=="ddim+pnp":
                    edited_image = edit_image_ddim_PnP(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                    )
                elif edit_method=="directinversion+pnp":
                    edited_image = edit_image_directinversion_PnP(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                    )
                else:
                    raise NotImplementedError(f"No edit method named {edit_method}")
                
                
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)
                
                print(f"finish")
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
        
        end_time = time.time()
        print(f"Delta time: {end_time - start_time:.2f} seconds")

        
if __name__ == "__main__":
    main()