# OSMIS Improved v2: full anatomy- and texture-guided one-shot synthesis

This repository keeps OSMIS as the one-shot GAN backbone, but separates four
sources of variation:

1. levator-hiatus mask shape;
2. large anatomy outside the hiatus;
3. visible structure inside the hiatus;
4. rendered-ultrasound texture.

The segmentation target remains the complete pear-shaped **levator-hiatus
interior** on the C-plane. Additional guidance changes how the whole image is
generated; it does not redefine the segmentation label.

## v2 model

```text
target hiatus mask ───────────────┐
                                  ├─ multi-scale SPADE ─┐
3-channel anatomy structure map ──┘                     │
                                                        ├─ image generator
reference image + reference mask ─ region style encoder ┤
                                  └─ high-res SEAN ─────┘
random noise ────────────────────────────────────────────┘

generated image ─┬─ conditional OSMIS discriminator
                 ├─ auxiliary hiatus segmentation head
                 ├─ structure consistency loss
                 └─ region texture consistency loss
```

### Structure condition

Every pseudo-pair includes a three-channel guide:

- low-frequency luminance for large anatomy;
- coarse anatomical edges;
- fine internal/external edges.

The discriminator sees image, target mask, and structure guide together.
Therefore changing only the hiatus mask is no longer the only route to
diversity.

### Texture condition

A lightweight SEAN-style encoder extracts separate style codes for hiatus and
background from a reference pseudo-image. High-resolution generator blocks use
these codes, while learned noise injection supplies stochastic local detail.

### Hierarchical pseudo-data

`prepare_anatomy_dataset.py` independently samples:

- a global smooth whole-image deformation;
- an internal residual field that vanishes at the hiatus boundary;
- an external residual field anchored at the hiatus and image borders;
- bounded gain, gamma, attenuation, blur, color, and correlated speckle.

Candidates are rejected when the levator-hiatus mask violates area, dimensions,
centroid, pear-shape, SP/PVM, connectivity, hole, or Jacobian constraints.
These are conservative within-state variations, not unsupported transitions
between rest, contraction, and Valsalva.

Before training, `validate_guidance_dataset.py` verifies that the generated
pseudo-data contain measurable variation both inside and outside the hiatus.

## Open-source references

- OSMIS: https://github.com/boschresearch/one-shot-synthesis
- SPADE reference: https://github.com/NVlabs/SPADE
- SEAN reference: https://github.com/ZPdesu/SEAN
- OASIS semantic discriminator: https://github.com/boschresearch/OASIS
- VoxelMorph deformation concepts: https://github.com/voxelmorph/voxelmorph

No source from SPADE's non-commercial repository is copied. The compact
SPADE/SEAN modules here are independently implemented from the published
formulas so the derived repository remains governed by the upstream OSMIS
AGPL-3.0 license.

The C-plane mask constraints follow:

- Sindhwani et al., *Semi-automatic outlining of levator hiatus*, UOG 2016.
- Bonmati et al., *Automatic segmentation method of pelvic floor levator
  hiatus in ultrasound using a self-normalizing neural network*, JMI 2018.

## RTX 5090 environment (recommended)

```bash
bash setup_5090.sh
conda activate osmis_5090
```

The setup installs a CUDA 12.8 PyTorch build. `verify_5090.py` checks CUDA,
compute capability, and an actual GPU matrix operation before training starts.

## RTX 5090 one-command training

The current RTX 5090 default pair is included in
`datasets/rendered_us_test2_source/`. The script synchronously removes the
top 20 rows from the image and mask to exclude the acquisition marker, then
uses conservative 2.8% global and 1.2% regional deformation amplitudes.

```bash
bash train_improved_5090.sh
```

Defaults:

- 64 validated pseudo-pairs;
- batch size 16 and 8 data-loader workers;
- 100,000 iterations;
- checkpoint, preview, and losses every 1,000 iterations;
- 3 high-resolution SEAN blocks.

Override without editing the script:

```bash
NUM_EPOCHS=10000 BATCH_SIZE=8 NUM_VARIANTS=64 bash train_improved_5090.sh my_test
```

Use another pair:

```bash
IMAGE_PATH=/path/image.png MASK_PATH=/path/mask.png bash train_improved_5090.sh my_case
```

Background launch, in one line:

```bash
nohup bash train_improved_5090.sh > train_stdout.log 2> train_stderr.log & echo $! > process_id.txt
```

If another process already occupies VRAM, reduce the batch without changing
the code: `BATCH_SIZE=8 bash train_improved_5090.sh`.

The original `train_improved.sh` remains the conservative batch-4 profile for
8 GB GPUs and local smoke testing.

## Generate

```bash
bash generate_improved.sh rendered_us_atg_osmis_full_v2 100000 50
```

Results:

```text
checkpoints/<experiment>/evaluation/<epoch>/
```

Each sample includes the image, target mask, auxiliary predicted mask, raw
labels, and the structure guide.

## Smoke test

```bash
bash run_smoke_test.sh
```

The two-step smoke test only verifies data generation, forward/backward,
checkpointing, and inference.

## Weight compatibility

v1 weights, including the previous 82,000-step SPADE-only checkpoint, are not
compatible with v2 because v2 adds structure channels, a region-style encoder,
SEAN blocks, and new discriminator inputs. Keep v1 weights as the baseline and
train v2 under a new experiment name.
