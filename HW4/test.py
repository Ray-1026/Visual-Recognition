import os
import random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.options import get_args
from utils.dataset import CustomTestDataset
from utils.model_utils import PromptIRModel

myseed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(myseed)
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)


def img2npz(folder_path, output_npz="pred.npz"):
    # Initialize dictionary to hold image arrays
    images_dict = {}

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(folder_path, filename)

            # Load image and convert to RGB
            image = Image.open(file_path).convert("RGB")
            img_array = np.array(image)

            # Rearrange to (3, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))

            # Add to dictionary
            images_dict[filename] = img_array

    # Save to .npz file
    np.savez(output_npz, **images_dict)

    print(f"Saved {len(images_dict)} images to {output_npz}")


def main():
    opt = get_args()

    if not opt.test_only:
        raise ValueError("Training is not supported in this script.")

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    test_dataset = CustomTestDataset(root_dir=opt.test_data_dir)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    model = PromptIRModel.load_from_checkpoint(opt.ckpt_path).to(opt.device)
    model.eval()

    with torch.no_grad():
        # for batch in test_dataloader:
        for batch in tqdm(test_dataloader, desc="Testing", total=len(test_dataloader)):
            [degrad_name], degrad_patch = batch
            degrad_patch = degrad_patch.to(opt.device)

            restored = model(degrad_patch)

            restored = restored.squeeze(0).cpu().numpy()
            restored = np.transpose(restored, (1, 2, 0))
            restored = np.clip(restored * 255, 0, 255).astype(np.uint8)
            restored_name = os.path.join(opt.output_dir, degrad_name[0])
            Image.fromarray(restored).save(restored_name)

    # Convert the output directory to .npz
    img2npz(opt.output_dir)

    # zip the npz file
    os.system("zip solution.zip pred.npz")


if __name__ == "__main__":
    main()
