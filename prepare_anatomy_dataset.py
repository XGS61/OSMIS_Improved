"""Create anatomy- and texture-diverse C-plane pseudo-pairs from one case.

The levator-hiatus mask remains the segmentation target.  Diversity outside
and inside that mask is introduced with separate smooth residual fields, while
a three-channel structure guide preserves the large anatomy and visible
internal edges.  All deformations are bounded and rejected if they fold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def infer_landmarks(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("The input mask is empty.")
    y_min, y_max = int(ys.min()), int(ys.max())
    band = max(2, int(round((y_max - y_min + 1) * 0.025)))
    return {
        "sp": [float(np.median(xs[ys <= y_min + band])), float(y_min)],
        "pvm": [float(np.median(xs[ys >= y_max - band])), float(y_max)],
    }


def row_width(mask, fraction):
    ys, _ = np.nonzero(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    y = int(round(y_min + fraction * (y_max - y_min)))
    band = mask[max(0, y - 2): min(mask.shape[0], y + 3)]
    _, xs = np.nonzero(band)
    return float(xs.max() - xs.min() + 1) if len(xs) else 0.0


def mask_metrics(mask):
    labels, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        mask = labels == int(np.argmax(sizes) + 1)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot measure an empty mask.")
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    upper_width = row_width(mask, 0.25)
    lower_width = row_width(mask, 0.75)
    landmarks = infer_landmarks(mask)
    return {
        "area": int(mask.sum()),
        "area_ratio_image": float(mask.mean()),
        "bbox_width": int(x_max - x_min + 1),
        "bbox_height": int(y_max - y_min + 1),
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "upper_width": upper_width,
        "lower_width": lower_width,
        "pear_ratio": float(upper_width / max(lower_width, 1.0)),
        "components": int(count),
        "sp": landmarks["sp"],
        "pvm": landmarks["pvm"],
    }


def make_smooth_field(shape, rng, max_dx, max_dy, grid_size):
    """Border-anchored cubic control-grid field."""
    h, w = shape
    ctrl_y = rng.normal(0.0, max_dy, size=(grid_size, grid_size))
    ctrl_x = rng.normal(0.0, max_dx, size=(grid_size, grid_size))
    for field in (ctrl_y, ctrl_x):
        field[[0, -1], :] = 0
        field[:, [0, -1]] = 0
    zoom = (h / grid_size, w / grid_size)
    field_y = ndimage.zoom(ctrl_y, zoom, order=3)[:h, :w]
    field_x = ndimage.zoom(ctrl_x, zoom, order=3)[:h, :w]
    sigma = max(h, w) / 45.0
    return (
        ndimage.gaussian_filter(field_y, sigma=sigma, mode="nearest"),
        ndimage.gaussian_filter(field_x, sigma=sigma, mode="nearest"),
    )


def minimum_jacobian(field_y, field_x):
    """Minimum determinant of T(y,x)=(y+dy,x+dx)."""
    dy_y, dy_x = np.gradient(field_y)
    dx_y, dx_x = np.gradient(field_x)
    determinant = (1.0 + dy_y) * (1.0 + dx_x) - dy_x * dx_y
    return float(determinant.min())


def warp(array, field_y, field_x, order):
    h, w = array.shape[:2]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = [yy + field_y, xx + field_x]
    if array.ndim == 2:
        return ndimage.map_coordinates(array, coords, order=order, mode="reflect")
    return np.stack(
        [
            ndimage.map_coordinates(array[..., channel], coords, order=order, mode="reflect")
            for channel in range(array.shape[2])
        ],
        axis=-1,
    )


def ratio(value, reference):
    return float(value) / max(float(reference), 1e-6)


def validate_mask(candidate, reference, constraints):
    labels, count = ndimage.label(candidate)
    if count != 1:
        return False, "not_single_component", None
    filled = ndimage.binary_fill_holes(candidate)
    hole_fraction = float((filled & ~candidate).sum()) / max(float(filled.sum()), 1.0)
    if hole_fraction > constraints["max_hole_fraction"]:
        return False, "holes", None
    metrics = mask_metrics(candidate)
    checks = {
        "area": constraints["area_ratio"][0]
        <= ratio(metrics["area"], reference["area"])
        <= constraints["area_ratio"][1],
        "width": constraints["width_ratio"][0]
        <= ratio(metrics["bbox_width"], reference["bbox_width"])
        <= constraints["width_ratio"][1],
        "height": constraints["height_ratio"][0]
        <= ratio(metrics["bbox_height"], reference["bbox_height"])
        <= constraints["height_ratio"][1],
        "centroid": np.hypot(
            metrics["centroid_x"] - reference["centroid_x"],
            metrics["centroid_y"] - reference["centroid_y"],
        )
        <= constraints["max_centroid_shift_px"],
        "pear": constraints["pear_ratio"][0]
        <= metrics["pear_ratio"]
        <= constraints["pear_ratio"][1],
        "sp": np.linalg.norm(np.asarray(metrics["sp"]) - np.asarray(reference["sp"]))
        <= constraints["max_landmark_shift_px"],
        "pvm": np.linalg.norm(np.asarray(metrics["pvm"]) - np.asarray(reference["pvm"]))
        <= constraints["max_landmark_shift_px"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, ",".join(failed), metrics


def border_taper(shape, width_fraction=0.10):
    h, w = shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    distance = np.minimum.reduce((yy, xx, h - 1 - yy, w - 1 - xx)).astype(np.float32)
    return np.clip(distance / max(2.0, min(h, w) * width_fraction), 0.0, 1.0)


def regional_residual_field(mask, rng, amplitude, grid_size):
    """Independent smooth changes inside and outside the hiatus.

    The residual vanishes at the hiatus boundary and image border, so it changes
    visible internal/external anatomy without breaking the supplied mask.
    """
    shape = mask.shape
    inner_y, inner_x = make_smooth_field(
        shape, rng, amplitude, amplitude, grid_size + 1
    )
    outer_y, outer_x = make_smooth_field(
        shape, rng, amplitude * 0.75, amplitude * 0.75, grid_size
    )
    inner_distance = ndimage.distance_transform_edt(mask)
    outer_distance = ndimage.distance_transform_edt(~mask)
    scale = max(2.0, min(shape) * 0.08)
    inner_weight = np.clip(inner_distance / scale, 0.0, 1.0) ** 0.75
    outer_weight = np.clip(outer_distance / scale, 0.0, 1.0) ** 0.75
    outer_weight *= border_taper(shape)
    field_y = inner_y * inner_weight + outer_y * outer_weight
    field_x = inner_x * inner_weight + outer_x * outer_weight
    return field_y, field_x


def robust_normalize(array, percentile=99.5):
    scale = float(np.percentile(array, percentile))
    if scale <= 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip(array / scale, 0.0, 1.0).astype(np.float32)


def structure_guide(image):
    """Low-frequency anatomy plus coarse and fine edge maps."""
    luminance = (
        0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    ) / 255.0
    low = ndimage.gaussian_filter(luminance, sigma=8.0)
    coarse = ndimage.gaussian_gradient_magnitude(luminance, sigma=3.0)
    fine = ndimage.gaussian_gradient_magnitude(luminance, sigma=1.2)
    return np.stack(
        (
            np.clip(low, 0.0, 1.0),
            robust_normalize(coarse),
            robust_normalize(fine),
        ),
        axis=-1,
    )


def apply_render_style(image, rng):
    """Bounded RGB rendering/speckle variation for the supplied orange volume."""
    normalized = np.clip(image / 255.0, 0.0, 1.0)
    gamma = float(rng.uniform(0.86, 1.16))
    gain = float(rng.uniform(0.90, 1.10))
    normalized = np.clip(normalized, 1e-5, 1.0) ** gamma
    normalized *= gain

    h, w = normalized.shape[:2]
    coarse = rng.normal(0.0, 1.0, size=(max(2, h // 24), max(2, w // 24)))
    bias = ndimage.zoom(coarse, (h / coarse.shape[0], w / coarse.shape[1]), order=3)[:h, :w]
    bias = ndimage.gaussian_filter(bias, sigma=max(h, w) / 60.0)
    bias /= max(float(np.std(bias)), 1e-6)
    normalized *= 1.0 + bias[..., None] * float(rng.uniform(0.0, 0.055))

    speckle = rng.normal(0.0, 1.0, size=(h, w))
    speckle = ndimage.gaussian_filter(speckle, sigma=float(rng.uniform(0.35, 1.0)))
    speckle /= max(float(np.std(speckle)), 1e-6)
    normalized *= 1.0 + speckle[..., None] * float(rng.uniform(0.0, 0.045))

    depth = np.linspace(1.0, float(rng.uniform(0.92, 1.04)), h)[:, None, None]
    normalized *= depth
    color = np.asarray(
        [rng.uniform(0.97, 1.04), rng.uniform(0.97, 1.03), rng.uniform(0.95, 1.04)]
    )
    normalized *= color[None, None, :]
    blur_sigma = float(rng.uniform(0.0, 0.55))
    if blur_sigma > 0.05:
        normalized = ndimage.gaussian_filter(normalized, sigma=(blur_sigma, blur_sigma, 0))
    return np.clip(normalized * 255.0, 0.0, 255.0), {
        "gamma": gamma,
        "gain": gain,
        "blur_sigma": blur_sigma,
    }


def save_pair(image, mask, structure, root, index):
    name = f"{index:05d}.png"
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB").save(
        root / "image" / name
    )
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(root / "mask" / name)
    Image.fromarray(np.clip(structure * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(
        root / "structure" / name
    )


def parse_point(value):
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Points must use x,y format.")
    return [float(parts[0]), float(parts[1])]


def main():
    parser = argparse.ArgumentParser(
        description="Generate full anatomy/texture-guided C-plane pseudo-pairs."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-variants", type=int, default=64)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--global-displacement-frac", type=float, default=0.040)
    parser.add_argument("--regional-displacement-frac", type=float, default=0.022)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--min-jacobian", type=float, default=0.35)
    parser.add_argument(
        "--crop-top",
        type=int,
        default=0,
        help="deterministically crop this many top rows from both image and mask",
    )
    parser.add_argument("--sp", type=parse_point)
    parser.add_argument("--pvm", type=parse_point)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32)
    mask = np.asarray(Image.open(args.mask).convert("L")) >= 128
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask mismatch: {image.shape[:2]} vs {mask.shape}")
    if args.crop_top < 0 or args.crop_top >= image.shape[0]:
        raise ValueError(
            f"--crop-top must be in [0, {image.shape[0] - 1}], got {args.crop_top}"
        )
    if args.crop_top:
        image = image[args.crop_top:, ...]
        mask = mask[args.crop_top:, ...]

    output = Path(args.output)
    if output.exists() and any(output.rglob("*.png")) and not args.overwrite:
        raise FileExistsError(f"{output} already contains PNG files; use --overwrite.")
    for subdir in ("image", "mask", "structure"):
        path = output / subdir
        path.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for old in path.glob("*.png"):
                old.unlink()

    reference = mask_metrics(mask)
    if args.sp is not None:
        reference["sp"] = args.sp
    if args.pvm is not None:
        reference["pvm"] = args.pvm
    min_dim = min(mask.shape)
    constraints = {
        "area_ratio": [0.82, 1.18],
        "width_ratio": [0.85, 1.15],
        "height_ratio": [0.91, 1.09],
        "pear_ratio": [
            max(1.30, reference["pear_ratio"] * 0.72),
            reference["pear_ratio"] * 1.35,
        ],
        "max_centroid_shift_px": float(min_dim * 0.045),
        "max_landmark_shift_px": float(min_dim * 0.055),
        "max_hole_fraction": 0.002,
        "min_jacobian": args.min_jacobian,
    }

    rng = np.random.default_rng(args.seed)
    records = []
    neutral_structure = structure_guide(image)
    save_pair(image, mask, neutral_structure, output, 0)
    records.append({"index": 0, "kind": "reference", "metrics": reference})

    attempts = 0
    while len(records) < args.num_variants and attempts < args.max_attempts:
        attempts += 1
        global_amp = min_dim * args.global_displacement_frac
        global_y, global_x = make_smooth_field(
            mask.shape,
            rng,
            max_dx=global_amp,
            max_dy=global_amp * 0.70,
            grid_size=args.grid_size,
        )
        global_jacobian = minimum_jacobian(global_y, global_x)
        if global_jacobian < args.min_jacobian:
            continue

        candidate = warp(mask.astype(np.float32), global_y, global_x, order=0) >= 0.5
        candidate = ndimage.binary_fill_holes(candidate)
        valid, _, metrics = validate_mask(candidate, reference, constraints)
        if not valid:
            continue

        geometric_image = warp(image, global_y, global_x, order=1)
        regional_amp = min_dim * args.regional_displacement_frac
        residual_y, residual_x = regional_residual_field(
            candidate, rng, regional_amp, args.grid_size
        )
        residual_jacobian = minimum_jacobian(residual_y, residual_x)
        if residual_jacobian < args.min_jacobian:
            continue
        geometric_image = warp(geometric_image, residual_y, residual_x, order=1)
        guide = structure_guide(geometric_image)
        styled_image, style_parameters = apply_render_style(geometric_image, rng)

        index = len(records)
        save_pair(styled_image, candidate, guide, output, index)
        records.append(
            {
                "index": index,
                "kind": "hierarchical_global_internal_external_warp",
                "metrics": metrics,
                "global_min_jacobian": global_jacobian,
                "regional_min_jacobian": residual_jacobian,
                "global_max_abs_x": float(np.abs(global_x).max()),
                "global_max_abs_y": float(np.abs(global_y).max()),
                "regional_max_abs_x": float(np.abs(residual_x).max()),
                "regional_max_abs_y": float(np.abs(residual_y).max()),
                "render_style": style_parameters,
            }
        )

    if len(records) < args.num_variants:
        raise RuntimeError(
            f"Generated {len(records)}/{args.num_variants} valid samples after "
            f"{attempts} attempts."
        )

    metadata = {
        "version": 2,
        "source_image": str(Path(args.image).resolve()),
        "source_mask": str(Path(args.mask).resolve()),
        "source_crop_top": args.crop_top,
        "segmentation_target": "levator-hiatus interior on the C-plane",
        "diversity": {
            "global": "smooth bounded whole-field deformation",
            "internal": "independent residual inside the hiatus, zero at boundary",
            "external": "independent residual outside the hiatus, border anchored",
            "texture": "bounded rendered-ultrasound photometric and speckle variation",
        },
        "structure_channels": [
            "low-frequency luminance",
            "coarse anatomical edge magnitude",
            "fine anatomical edge magnitude",
        ],
        "clinical_scope": (
            "Within-state variation only; no unsupported rest/contraction/Valsalva transition."
        ),
        "constraints": constraints,
        "reference_metrics": reference,
        "accepted": len(records),
        "attempts": attempts,
        "samples": records,
    }
    with open(output / "anatomy_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"Prepared {len(records)} full-guidance pairs at {output}")
    print(f"Accepted {len(records) - 1} variants from {attempts} attempts")
    print(f"SP={reference['sp']}, PVM={reference['pvm']}")


if __name__ == "__main__":
    main()
