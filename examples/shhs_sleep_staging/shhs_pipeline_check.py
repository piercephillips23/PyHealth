"""Verify SHHS EDF/XML preprocessing, masking, and PyHealth MLP integration.

Place beside shhs_mlp.py in the cloned PyHealth/examples directory.
Run from the repository root (PowerShell):

python examples/shhs_pipeline_check.py `
  --edf-dir "C:\\Users\\potan\\shhs\\polysomnography\\edfs\\shhs1" `
  --xml-dir "C:\\Users\\potan\\shhs\\polysomnography\\annotations-events-nsrr\\shhs1" `
  --recording-id shhs1-200005

This is a pipeline integration check, NOT a scientific evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from pyhealth.tasks.shhs_multimodal_masking import (
    apply_modality_mask,
    collate_shhs_epochs,
)
from pyhealth.tasks.shhs_multimodal_sleep_staging import SHHSMultimodalSleepStaging
from shhs_mlp import PyHealthMaskedMLP


SUBSETS = (
    ("ecg",),
    ("abd", "thx"),
    ("ecg", "thx"),
    ("ecg", "abd", "thx"),
)
EXPECTED_LENGTHS = {"ecg": 1024, "abd": 256, "thx": 256}


def on_device(batch: dict, device: torch.device) -> dict:
    """Move tensors only; keep patient/recording metadata as Python lists."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def check_batch(batch: dict, n: int) -> None:
    for name, length in EXPECTED_LENGTHS.items():
        assert tuple(batch[name].shape) == (n, length), (name, batch[name].shape)
        assert bool(torch.isfinite(batch[name]).all()), f"Non-finite {name} values"
    assert tuple(batch["availability_mask"].shape) == (n, 3)
    assert tuple(batch["input_mask"].shape) == (n, 3)
    assert tuple(batch["label"].shape) == (n,)
    available = batch["availability_mask"].bool()
    selected = batch["input_mask"].bool()
    assert not bool((selected & ~available).any()), "Selected unavailable modality"
    assert bool(selected.any(dim=1).all()), "All modalities masked for an epoch"
    assert bool(((batch["label"] >= 0) & (batch["label"] <= 3)).all())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf-dir", type=Path, required=True)
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--recording-id", default="shhs1-200005")
    args = parser.parse_args()

    edf = args.edf_dir / f"{args.recording_id}.edf"
    xml = args.xml_dir / f"{args.recording_id}-nsrr.xml"
    if not edf.is_file() or not xml.is_file():
        parser.error(f"Missing matched EDF/XML: {edf} | {xml}")

    torch.manual_seed(6250)
    task = SHHSMultimodalSleepStaging()
    samples = task.load_recording(str(edf), str(xml), args.recording_id)
    assert samples, "Loader returned no samples"
    counts = Counter(int(sample["label"]) for sample in samples)
    print(f"SHHS recording: {args.recording_id}; epochs={len(samples)}; labels={dict(sorted(counts.items()))}")

    # Deliberately include all four classes to test multiclass bookkeeping,
    # rather than selecting eight adjacent epochs that may all be Wake.
    selected_samples = []
    for label in range(4):
        chosen = [sample for sample in samples if int(sample["label"]) == label][:2]
        if len(chosen) < 2:
            raise RuntimeError(f"Need two scored epochs for class {label}; found {len(chosen)}")
        selected_samples.extend(chosen)

    batch = collate_shhs_epochs(selected_samples)
    batch = apply_modality_mask(batch, policy="none", rng=np.random.default_rng(6250))
    check_batch(batch, n=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyHealthMaskedMLP(hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_batch = on_device(batch, device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(train_batch)
    assert tuple(output["logit"].shape) == (8, 4)
    assert bool(torch.isfinite(output["loss"]))
    output["loss"].backward()
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ), "No finite parameter gradient"
    optimizer.step()
    print(
        f"PASS: EDF/XML -> 8 epoch batch -> PyHealth MLP -> "
        f"finite loss ({output['loss'].item():.4f}) -> optimizer step ({device})"
    )

    # Use the project's masking policy implementation, not the provisional sampler.
    model.eval()
    for policy in ("none", "paper", "balanced"):
        masked = apply_modality_mask(batch, policy=policy, rng=np.random.default_rng(6250))
        check_batch(masked, n=8)
        with torch.no_grad():
            prediction = model(on_device(masked, device))["logit"]
        assert tuple(prediction.shape) == (8, 4)
        print(f"PASS: masking policy '{policy}' -> logits (8, 4)")

    for subset in SUBSETS:
        subset_batch = apply_modality_mask(
            batch,
            policy="eval",
            requested_subset=subset,
        )
        check_batch(subset_batch, n=8)
        with torch.no_grad():
            prediction = model(on_device(subset_batch, device))["logit"]
        assert tuple(prediction.shape) == (8, 4)
        print(f"PASS: evaluation subset {subset} -> logits (8, 4)")

    print("PASS: SHHS loader + collator + masking + PyHealth MLP integration")
    print("NOTE: This is an integration check, not a train/validation/test experiment.")


if __name__ == "__main__":
    main()
