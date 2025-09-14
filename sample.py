import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from Utils.viz import show_images_grid
from huggingface_hub import hf_hub_download
from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler


def to_uint8_nhwc(images: torch.Tensor) -> np.ndarray:
    """Convert a batch of images in [-1, 1] (B, C, H, W) to uint8 NHWC numpy."""
    images = ((images + 1.0) / 2.0).clamp(0, 1)
    images = (images * 255.0).round().to(torch.uint8).cpu().numpy()
    images = images.transpose(0, 2, 3, 1)
    return images


@torch.no_grad()
def sample_and_save(cli_args):
    args = load_args_from_file(cli_args.config)

    # Update arguments
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select Network (Large 384 is the best, but the slowest)
    args.vit_size = "tiny"  # "tiny", "small", "base", "large"
    args.img_size = 256
    args.compile = True
    args.dtype = "bfloat16"
    args.resume = True
    args.is_master = True

    args.sampler = "halton"
    args.sched_mode = 'arccos'
    args.cfg_w = 1.5
    args.r_temp = 5.0
    args.sched_pow=2
    args.top_k=-1
    args.temp_warmup=1
    args.sm_temp_min=1.
    args.sm_temp=1.0
    args.step=32

    args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"

    # Download the MaskGIT
    hf_hub_download(repo_id="llvictorll/Halton-Maskgit",
                    filename=f"ImageNet_{args.img_size}_{args.vit_size}.pth",
                    local_dir="./saved_networks")

    # Download VQGAN
    hf_hub_download(repo_id="FoundationVision/LlamaGen",
                    filename="vq_ds16_c2i.pt",
                    local_dir="./saved_networks")


    # Initialisation of the model
    model = MaskGIT(args)

    # select your scheduler (Halton is better)
    sampler = model.sampler

    total = cli_args.num_images
    num_classes = cli_args.num_classes
    assert total % num_classes == 0, "num_images must be divisible by num_classes"
    per_class = total // num_classes

    # Prepare output array (uint8, NHWC)
    height = args.img_size
    width = args.img_size
    out_images = np.zeros((total, height, width, 3), dtype=np.uint8)
    out_labels = np.zeros((total,), dtype=np.int64)

    # Derive model name from vit_folder
    vit_folder = args.vit_folder
    model_name = os.path.splitext(os.path.basename(vit_folder))[0]

    os.makedirs(cli_args.output_dir, exist_ok=True)
    out_path = os.path.join(cli_args.output_dir, f"{model_name}.npz")

    # 2) Build label schedule: repeat each class equally
    all_labels = torch.arange(num_classes, dtype=torch.long).repeat_interleave(per_class)

    # 3) Generate in batches
    batch_size = cli_args.batch_size
    num_batches = (total + batch_size - 1) // batch_size
    start = 0
    pbar = tqdm(range(num_batches), desc="Sampling", leave=True)
    for _ in pbar:
        end = min(start + batch_size, total)
        labels = all_labels[start:end].to(args.device)
        nb = labels.size(0)

        generated = sampler(trainer=model, nb_sample=nb, labels=labels, verbose=False)[0]
        out_images[start:end] = to_uint8_nhwc(generated)
        out_labels[start:end] = labels.cpu().numpy()

        start = end

    # 4) Save as NPZ: images at arr_0 (NHWC), labels at arr_1
    np.savez_compressed(out_path, out_images, out_labels)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Config/base_cls2img.yaml", help="YAML config path")
    parser.add_argument("--output-dir", type=str, default="saved_datasets", help="Destination directory")
    parser.add_argument("--num-images", type=int, default=50_000, help="Total number of images to sample")
    parser.add_argument("--num-classes", type=int, default=1_000, help="Number of classes to cover")
    parser.add_argument("--batch-size", type=int, default=256, help="Sampling batch size")
    args = parser.parse_args()

    sample_and_save(args)


if __name__ == "__main__":
    main()
