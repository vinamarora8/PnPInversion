import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import json


class LaionFolderDataset(Dataset):
    def __init__(self, root_dir, image_size=512):
        self.root_dir = root_dir

        # Find all .jpg files inside all subfolders (00000, 00001, etc.)
        # This might take a moment to scan 1 million files
        print(f"Scanning {root_dir} for images...")
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*", "*.jpg")))
        print(f"Found {len(self.image_paths)} images.")

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # 1. Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Handle corrupt images by returning a blank tensor or skipping
            # (In a real training loop, you'd usually catch this in collate_fn)
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 512, 512), "error"

        image = self.transform(image)

        # 2. Load Caption (Optional)
        # Assuming metadata is saved as same filename but .json or .txt
        # e.g. 000000000.jpg -> 000000000.json
        caption = ""
        json_path = img_path.replace(".jpg", ".json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    meta = json.load(f)
                    caption = meta.get("caption", "") or meta.get("TEXT", "")
            except:
                pass

        return image, caption


if __name__ == "__main__":
    dataset = LaionFolderDataset("laion2B-en-aesthetic/laion_100gb_512px/")
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    for images, captions in loader:
        print(f"Batch shape: {images.shape}")
        print(captions)
        break
