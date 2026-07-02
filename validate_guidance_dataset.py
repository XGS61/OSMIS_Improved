"""Report whether pseudo-data vary inside, outside, in mask, and in structure."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_stack(paths, mode):
    return np.stack(
        [np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0 for path in paths]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--min-image-std", type=float, default=0.01)
    args = parser.parse_args()

    root = Path(args.dataset)
    image_paths = sorted((root / "image").glob("*.png"))
    mask_paths = sorted((root / "mask").glob("*.png"))
    structure_paths = sorted((root / "structure").glob("*.png"))
    if not image_paths or not (
        len(image_paths) == len(mask_paths) == len(structure_paths)
    ):
        raise RuntimeError("Incomplete image/mask/structure pseudo-dataset.")

    images = load_stack(image_paths, "RGB").mean(axis=-1)
    masks = load_stack(mask_paths, "L") > 0.5
    structures = load_stack(structure_paths, "RGB")
    reference_mask = masks[0]
    temporal_std = images.std(axis=0)
    inside_std = float(temporal_std[reference_mask].mean())
    outside_std = float(temporal_std[~reference_mask].mean())
    mask_area = masks.mean(axis=(1, 2))
    structure_std = structures.std(axis=0).mean(axis=(0, 1))

    print(f"samples={len(images)}")
    print(f"mask_area_mean={mask_area.mean():.6f}")
    print(f"mask_area_std={mask_area.std():.6f}")
    print(f"image_std_inside={inside_std:.6f}")
    print(f"image_std_outside={outside_std:.6f}")
    print(
        "structure_channel_std="
        + ",".join(f"{value:.6f}" for value in structure_std)
    )
    if inside_std < args.min_image_std or outside_std < args.min_image_std:
        raise RuntimeError(
            "Pseudo-data lack variation inside or outside the hiatus; "
            "review deformation limits before training."
        )


if __name__ == "__main__":
    main()
