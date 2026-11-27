"""Verify SHHS EDF/XML preprocessing, masking, and PyHealth MLP integration.

# Public demo: requires no SHHS recordings.
    python examples/shhs_sleep_staging/shhs_pipeline_check.py --synthetic

    # Real-data integration check.
    python examples/shhs_sleep_staging/shhs_pipeline_check.py \
      --edf-dir "<path-to-shhs1-edfs>" \
      --xml-dir "<path-to-shhs1-annotations>" \
      --recording-id shhs1-200005

Both modes check a single optimizer step and fixed modality subsets; neither is
an actual train/validation/test performance experiment. Synthetic signals are
random fixtures, not physiologically realistic sleep recordings.
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
from shhs_mlp import PyHealthMaskedMLP


SUBSETS = (
    ("ecg",),
    ("abd", "thx"),
    ("ecg", "thx"),
    ("ecg", "abd", "thx"),
)
EXPECTED_LENGTHS = {"ecg": 1024, "abd": 256, "thx": 256}


def make_synthetic_samples(seed: int = 6250) -> list[dict]:
    """Generate eight contract-compatible epochs: two from each stage.

    These are synthetic, in-memory examples only; no EDF or XML is read.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for index in range(8):
        samples.append(
            {
                "ecg": rng.normal(size=1024).astype(np.float32),
                "abd": rng.normal(size=256).astype(np.float32),
                "thx": rng.normal(size=256).astype(np.float32),
                "availability_mask": np.array([True, True, True], dtype=np.bool_),
                "label": index % 4,
                "patient_id": "synthetic-patient-001",
                "recording_id": "synthetic-recording-001",
                "epoch_index": index,
            }
        )
    return samples


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


def real_samples(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[dict]:
    if args.edf_dir is None or args.xml_dir is None:
        parser.error("Real-data mode requires --edf-dir and --xml-dir (or use --synthetic).")

    edf = args.edf_dir / f"{args.recording_id}.edf"
    xml = args.xml_dir / f"{args.recording_id}-nsrr.xml"
    if not edf.is_file() or not xml.is_file():
        parser.error(f"Missing matched EDF/XML: {edf} | {xml}")

    # Do not import the real EDF/XML task when using the synthetic-only demo.
    from pyhealth.tasks.shhs_multimodal_sleep_staging import SHHSMultimodalSleepStaging

    task = SHHSMultimodalSleepStaging()
    samples = task.load_recording(str(edf), str(xml), args.recording_id)
    assert samples, "Loader returned no samples"
    counts = Counter(int(sample["label"]) for sample in samples)
    print(
        f"SHHS recording: {args.recording_id}; "
        f"epochs={len(samples)}; labels={dict(sorted(counts.items()))}"
    )

    # Include all four stages rather than eight adjacent, possibly same-stage epochs.
    selected_samples = []
    for label in range(4):
        chosen = [sample for sample in samples if int(sample["label"]) == label][:2]
        if len(chosen) < 2:
            raise RuntimeError(f"Need two scored epochs for class {label}; found {len(chosen)}")
        selected_samples.extend(chosen)
    return selected_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic", action="store_true", help="Run on generated epochs; no EDF/XML required"
    )
    parser.add_argument("--edf-dir", type=Path)
    parser.add_argument("--xml-dir", type=Path)
    parser.add_argument("--recording-id", default="shhs1-200005")
    args = parser.parse_args()

    torch.manual_seed(6250)
    if args.synthetic:
        samples = make_synthetic_samples(seed=6250)
        counts = Counter(int(sample["label"]) for sample in samples)
        print(f"Synthetic example: epochs={len(samples)}; labels={dict(sorted(counts.items()))}")
    else:
        samples = real_samples(args, parser)

    batch = collate_shhs_epochs(samples)
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
    source = "Synthetic epochs" if args.synthetic else "EDF/XML"
    print(
        f"PASS: {source} -> 8 epoch batch -> PyHealth MLP -> "
        f"finite loss ({output['loss'].item():.4f}) -> optimizer step ({device})"
    )

    # Use the team's actual masking policies for synthetic AND real-data modes.
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

    if args.synthetic:
        print("PASS: Synthetic batch + collator + masking + PyHealth MLP integration")
    else:
        print("PASS: SHHS loader + collator + masking + PyHealth MLP integration")