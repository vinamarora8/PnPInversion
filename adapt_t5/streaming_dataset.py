import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from torchvision import transforms


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


class LaionStreamingDataset(IterableDataset):
    def __init__(self, dataset_name, split="train", buffer_size=1000):
        self.buffer_size = buffer_size

        # 1. Load the dataset in streaming mode (no download)
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)

        # 2. Define image transforms (Resize to 512x512 for Stable Diffusion)
        self.transform = transforms.Compose(
            [
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # Map to [-1, 1] for Diffusion
            ]
        )

    def __iter__(self):
        # This generator yields (image_tensor, text_caption) pairs
        for sample in self.dataset:
            try:
                # Extract data
                # image = sample.get('image') or sample.get('jpg') # COCO uses 'image', LAION uses 'jpg'
                url = sample.get("URL")
                if url is None:
                    continue
                import requests
                from io import BytesIO

                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                text = sample.get("TEXT")

                # Basic validation
                if image is None or text is None:
                    continue

                # Apply transforms
                image_tensor = self.transform(image.convert("RGB"))

                yield image_tensor, text

            except Exception as e:
                # Skip broken images (common in streaming datasets)
                continue
