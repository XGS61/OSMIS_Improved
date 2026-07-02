#!/usr/bin/env python3
"""Fail fast when the active PyTorch environment cannot use an RTX 5090."""

from __future__ import annotations

import sys

import torch


def version_tuple(version: str | None) -> tuple[int, int]:
    if not version:
        return (0, 0)
    fields = version.split(".")
    return int(fields[0]), int(fields[1])


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False.", file=sys.stderr)
        print("Install the CUDA 12.8 PyTorch build with bash setup_5090.sh.", file=sys.stderr)
        return 1

    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    cuda_version = torch.version.cuda

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {cuda_version}")
    print(f"GPU: {name}")
    print(f"Compute capability: sm_{capability[0]}{capability[1]}")

    if capability >= (12, 0) and version_tuple(cuda_version) < (12, 8):
        print(
            "ERROR: Blackwell requires a PyTorch binary built with CUDA 12.8 or newer.",
            file=sys.stderr,
        )
        return 2

    probe = torch.randn(1024, 1024, device="cuda")
    value = (probe @ probe.T).mean()
    torch.cuda.synchronize()
    print(f"CUDA matrix test: OK ({value.item():.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
