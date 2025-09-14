#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def _to_uint8(img: np.ndarray) -> np.ndarray:
  if img.dtype == np.uint8:
    arr = img
  else:
    arr = np.clip(img, 0, 255).astype(np.uint8)
  if arr.ndim == 2:
    arr = np.repeat(arr[..., None], 3, axis=2)
  if arr.shape[-1] == 1:
    arr = np.repeat(arr, 3, axis=-1)
  return arr


def _build_grid(imgs: List[np.ndarray], rows: int = 10, cols: int = 10) -> np.ndarray:
  if len(imgs) == 0:
    raise ValueError('No images to build grid.')

  h, w = imgs[0].shape[:2]
  canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

  total = rows * cols
  for k in range(total):
    r = k // cols
    c = k % cols
    y0 = r * h
    x0 = c * w
    if k < len(imgs):
      tile = imgs[k]
      tile = _to_uint8(tile)
      if tile.shape[:2] != (h, w):
        raise ValueError('All images must share same HxW to build grid.')
    else:
      tile = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[y0:y0 + h, x0:x0 + w] = tile
  return canvas


def parse_args():
  parser = argparse.ArgumentParser(description='Sample a 10x10 grid of images from an NPZ file.')
  parser.add_argument('--npz', required=True, help='Path to NPZ file containing images under a given array name.')
  parser.add_argument('--arr', default='arr_0', help="Array name inside NPZ (default: 'arr_0')")
  parser.add_argument('--outdir', default='.', help='Directory to save the output grid image (default: current directory).')
  parser.add_argument('--grid_name', default='grid_10x10.png', help='Output filename for the grid image.')
  parser.add_argument('--rows', type=int, default=10, help='Number of rows in the grid (default: 10).')
  parser.add_argument('--cols', type=int, default=10, help='Number of columns in the grid (default: 10).')
  parser.add_argument('--seed', type=int, default=0, help='Random seed for sampling (default: 0).')
  parser.add_argument('--strategy', choices=['random', 'first'], default='random',
                      help="Sampling strategy: 'random' to sample without replacement, 'first' to take first N (default: random).")
  return parser.parse_args()


def main():
  args = parse_args()
  npz_path = args.npz
  outdir = Path(args.outdir)
  total_needed = args.rows * args.cols

  if not os.path.isfile(npz_path):
    raise FileNotFoundError(f'NPZ not found: {npz_path}')

  with np.load(npz_path) as data:
    if args.arr not in data:
      raise KeyError(f"Array '{args.arr}' not found in NPZ. Available: {list(data.keys())}")
    arr = data[args.arr]

  if arr.ndim < 3:
    raise ValueError('Expected array with shape [N, H, W, C] or [N, H, W].')
  if arr.ndim == 3:
    # [N, H, W] -> assume grayscale
    pass
  elif arr.ndim == 4 and arr.shape[-1] not in (1, 3):
    raise ValueError('Expected channel dimension to be 1 or 3 when arr is 4D.')

  num = arr.shape[0]
  if num == 0:
    raise ValueError('Empty array: no images to visualize.')

  if args.strategy == 'random':
    rng = np.random.default_rng(args.seed)
    if num >= total_needed:
      indices = rng.choice(num, size=total_needed, replace=False)
    else:
      indices = rng.choice(num, size=total_needed, replace=True)
  else:
    indices = np.arange(min(total_needed, num))

  sampled = [arr[i] for i in indices]
  # Convert and ensure consistent HxW
  sampled_uint8 = [_to_uint8(x) for x in sampled]
  target_h, target_w = sampled_uint8[0].shape[:2]
  for im in sampled_uint8:
    if im.shape[:2] != (target_h, target_w):
      raise ValueError('Images with varying sizes are not supported.')

  grid = _build_grid(sampled_uint8, rows=args.rows, cols=args.cols)
  outdir.mkdir(parents=True, exist_ok=True)
  grid_path = outdir / args.grid_name
  Image.fromarray(grid).save(str(grid_path))
  print(f'Wrote grid image to {grid_path}')


if __name__ == '__main__':
  main()


