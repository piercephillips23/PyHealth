"""Reproducible SHHS1 multimodal sleep-staging masking ablation.

The script uses Person 1's SHHS task, collator and mask sampler, plus the
existing PyHealth MLP through shhs_mlp.py. This is a single-epoch MLP baseline,
NOT a reproduction of wav2sleep's CNN/Transformer/temporal architecture.

Run from the cloned PyHealth repository root (PowerShell):

    python examples/shhs_ablation.py `
      --manifest "C:\\Users\\potan\\shhs\\shhs1_manifest.csv" `
      --cache-dir "C:\\Users\\potan\\shhs\\processed_cache" `
      --output-dir "C:\\Users\\potan\\shhs\\analysis_3ep" `
      --max-epochs-per-recording 120 --epochs 3

Three training policies (none, paper, balanced) share one participant-disjoint
manifest split, fixed data sampling and fixed test modality subsets.
Checkpoints are selected by validation mean Cohen's kappa, never by test data.

Exports summary and per-subset results, validation history, selected-checkpoint
validation/test predictions (with real participant/recording/epoch IDs), class
precision/recall/F1, and confusion matrices. Rows of confusion matrices have
true labels; columns have predicted labels.

IMPORTANT: EDF files, processed caches, prediction-level exports and local data
paths should remain outside the Git repository / final upstream PyHealth PR.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import zlib
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from pyhealth.tasks.shhs_multimodal_masking import (
    apply_modality_mask,
    collate_shhs_epochs,
    sample_input_mask,
)
from pyhealth.tasks.shhs_multimodal_sleep_staging import SHHSMultimodalSleepStaging
from shhs_mlp import CLASS_NAMES, PyHealthMaskedMLP

MODALITIES = ("ecg", "abd", "thx")
SIGNAL_LENGTHS = {"ecg": 1024, "abd": 256, "thx": 256}
CONDITIONS = ("none", "paper", "balanced")
SUBSETS = (("ecg",), ("abd", "thx"), ("ecg", "thx"), MODALITIES)
SPLITS = ("train", "val", "test")
N_CLASSES = len(CLASS_NAMES)


class CachedRecording(NamedTuple):
    """Manifest identity accompanies the cached epoch arrays, not fake IDs."""

    recording_id: str
    participant_id: str
    path: Path



def read_manifest(path: Path, limit_per_split: int) -> dict[str, list[dict[str, str]]]:
    """Preserve the frozen manifest splits; optionally select a small cohort."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    required = {"recording_id", "participant_id", "edf_path", "xml_path", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest needs columns {sorted(required)}")
    groups: dict[str, list[dict[str, str]]] = {part: [] for part in SPLITS}
    owners: dict[str, str] = {}
    seen_ids: set[str] = set()
    for row in rows:
        split = row["split"].strip().lower()
        if split not in groups:
            raise ValueError(f"Unrecognized split {split!r}")
        pid = row["participant_id"].strip()
        rid = row["recording_id"].strip()
        if not pid or not rid or rid in seen_ids:
            raise ValueError(f"Missing or duplicate recording ID: {rid!r}")
        seen_ids.add(rid)
        if pid in owners and owners[pid] != split:
            raise ValueError(f"Participant leakage across splits: {pid}")
        owners[pid] = split
        groups[split].append(row)
    for split in SPLITS:
        groups[split].sort(key=lambda row: row["recording_id"])
        if limit_per_split:
            # Count distinct participants, not epochs. SHHS1 generally has one
            # recording per participant, but this also guards future changes.
            selected: list[dict[str, str]] = []
            pids: set[str] = set()
            for row in groups[split]:
                if row["participant_id"] in pids or len(pids) < limit_per_split:
                    pids.add(row["participant_id"])
                    selected.append(row)
            groups[split] = selected
        if not groups[split]:
            raise ValueError(f"No recordings in {split} split")
    return groups


def cache_path(cache_dir: Path, rid: str, max_epochs: int) -> Path:
    if not rid.startswith("shhs1-") or not rid[len("shhs1-"):].isdigit():
        raise ValueError(f"Unexpected recording ID: {rid!r}")
    suffix = f"first{max_epochs}" if max_epochs else "all"
    return cache_dir / f"{rid}_{suffix}.npz"


def make_cache(
    row: dict[str, str], cache_dir: Path, max_epochs: int,
    task: SHHSMultimodalSleepStaging,
) -> tuple[Path, int, Counter]:
    """Preprocess each recording once; preserve a fixed complete-case cohort."""
    rid = row["recording_id"]
    path = cache_path(cache_dir, rid, max_epochs)
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            labels = data["label"]
            count = len(labels)
            return path, count, Counter(map(int, labels.tolist()))
    samples = task.load_recording(
        row["edf_path"], row["xml_path"], row["participant_id"]
    )
    if not samples:
        raise ValueError("Loader returned zero epochs")
    if max_epochs and len(samples) > max_epochs:
        # Deterministic distributed subsample, not only the first part of the
        # night; avoids systematic stride aliasing with stage transitions.
        per_record_seed = 6250 + zlib.crc32(rid.encode("utf-8"))
        indices = np.random.default_rng(per_record_seed).choice(
            len(samples), size=max_epochs, replace=False
        )
        samples = [samples[int(i)] for i in sorted(indices)]
    usable = []
    for s in samples:
        label = int(s["label"])
        availability = np.asarray(s["availability_mask"], dtype=np.bool_)
        if not 0 <= label <= 3:
            continue  # Unscored labels may not be passed to cross-entropy.
        if availability.shape != (3,):
            raise ValueError(f"{rid}: bad availability mask shape")
        # The 4 fixed test subsets must have the same subjects/epochs.
        # Strict complete-case cohort ensures that; report any exclusions.
        if not bool(availability.all()):
            continue
        for name in MODALITIES:
            signal = np.asarray(s[name])
            if signal.shape != (SIGNAL_LENGTHS[name],) or not np.isfinite(signal).all():
                raise ValueError(f"{rid}: invalid signal {name} {signal.shape}")
        usable.append(s)
    if not usable:
        raise ValueError("No complete-case labeled epochs")
    arrays = {
        name: np.stack([np.asarray(s[name], dtype=np.float32) for s in usable])
        for name in MODALITIES
    }
    arrays["label"] = np.asarray([int(s["label"]) for s in usable], dtype=np.int64)
    arrays["epoch_index"] = np.asarray(
        [int(s["epoch_index"]) for s in usable], dtype=np.int32
    )
    arrays["availability_mask"] = np.stack(
        [np.asarray(s["availability_mask"], dtype=np.bool_) for s in usable]
    )
    # Writing to a temporary file avoids mistaking an interrupted cache for
    # a valid file in later attempts.
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return path, len(usable), Counter(map(int, arrays["label"].tolist()))


def preprocess(
    groups: dict[str, list[dict[str, str]]], cache_dir: Path, max_epochs: int
) -> tuple[dict[str, list[CachedRecording]], dict[str, dict[str, int]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    task = SHHSMultimodalSleepStaging()
    prepared: dict[str, list[CachedRecording]] = {part: [] for part in SPLITS}
    stats = {}
    for split, rows in groups.items():
        total_labels = Counter()
        skipped = []
        for idx, row in enumerate(rows, start=1):
            try:
                path, count, labels = make_cache(row, cache_dir, max_epochs, task)
                prepared[split].append(CachedRecording(
                    recording_id=row["recording_id"],
                    participant_id=row["participant_id"],
                    path=path,
                ))
                total_labels.update(labels)
            except (OSError, ValueError, KeyError, RuntimeError) as err:
                skipped.append((row["recording_id"], str(err)))
            if idx % 10 == 0 or idx == len(rows):
                print(f"  {split}: preprocessed {idx}/{len(rows)} recordings", flush=True)
        if skipped:
            print(f"WARNING: {len(skipped)} {split} recordings excluded: {skipped[:8]}")
        if not prepared[split]:
            raise ValueError(f"All {split} recordings were excluded")
        stats[split] = {
            "n_recordings": len(prepared[split]),
            "n_epochs": sum(total_labels.values()),
            **{f"class_{c}": total_labels[c] for c in range(4)},
            "excluded_recordings": len(skipped),
        }
        print(f"{split} cohort: {stats[split]}")
    return prepared, stats


def to_batch(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    participant_id: str,
    recording_id: str,
) -> dict:
    """Collate cached epochs while preserving genuine manifest identifiers."""
    samples = []
    for ix in indices:
        samples.append({
            **{name: data[name][int(ix)] for name in MODALITIES},
            "availability_mask": data["availability_mask"][int(ix)],
            "label": int(data["label"][int(ix)]),
            "epoch_index": int(data["epoch_index"][int(ix)]),
            "patient_id": participant_id,
            "recording_id": recording_id,
        })
    return collate_shhs_epochs(samples)


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        name: value.to(device) if isinstance(value, Tensor) else value
        for name, value in batch.items()
    }


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    """Compute four-class accuracy, unweighted kappa, and macro F1.

    Confusion axes: [true class, predicted class]. Absent classes contribute
    F1=0 to the four-class macro average, and undefined precision/recall are 0.
    """
    if confusion.shape != (N_CLASSES, N_CLASSES):
        raise ValueError(f"Confusion must be [{N_CLASSES},{N_CLASSES}]")
    if np.any(confusion < 0):
        raise ValueError("Confusion counts must be nonnegative")
    support = confusion.sum(axis=1).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    n = float(support.sum())
    if n == 0:
        raise ValueError("Cannot compute metrics with zero observations")
    tp = np.diag(confusion).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    accuracy = float(tp.sum() / n)
    expected = float(np.dot(predicted, support) / (n * n))
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 - 1e-12 else float("nan")
    return {"accuracy": accuracy, "kappa": kappa, "macro_f1": float(f1.mean())}


def class_metrics_from_confusion(confusion: np.ndarray) -> list[dict[str, float | int | str]]:
    """Return per-class support, predicted count, precision, recall, and F1."""
    reports = []
    for cls, name in enumerate(CLASS_NAMES):
        tp = int(confusion[cls, cls])
        support = int(confusion[cls, :].sum())
        predicted = int(confusion[:, cls].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        reports.append({
            "class_id": cls,
            "class_name": name,
            "support": support,
            "predicted_count": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return reports


def train_epoch(
    model: PyHealthMaskedMLP, optimizer: torch.optim.Optimizer,
    items: list[CachedRecording], condition: str, epoch: int,
    batch_size: int, seed: int, device: torch.device,
) -> float:
    model.train()
    py_rng = random.Random(seed + epoch * 1009)
    np_rng = np.random.default_rng(seed + epoch * 1009)
    order = items.copy()
    py_rng.shuffle(order)
    total = 0.0
    count = 0
    for item in order:
        rid = item.recording_id
        with np.load(item.path, allow_pickle=False) as archived:
            data = {key: archived[key] for key in (*MODALITIES, "label", "epoch_index", "availability_mask")}
            n = len(data["label"])
            indices = np_rng.permutation(n)
            # One training mask per recording per training epoch. The same
            # natural modality availability is required for all included epochs.
            avail = np.asarray(data["availability_mask"][0], dtype=np.bool_)
            if not np.all(data["availability_mask"] == avail):
                raise ValueError(f"{rid}: availability differs between epochs")
            selected = np.asarray(
                sample_input_mask(avail, policy=condition, rng=np_rng),
                dtype=np.bool_,
            )
            if selected.shape != (3,) or not selected.any() or (selected & ~avail).any():
                raise ValueError(f"Invalid sampled mask for {rid}: {selected}")
            for start in range(0, n, batch_size):
                batch = to_batch(
                    data, indices[start : start + batch_size],
                    item.participant_id, item.recording_id,
                )
                batch["input_mask"] = torch.as_tensor(selected).expand(
                    len(batch["label"]), 3
                ).clone()
                batch = to_device(batch, device)
                result = model(batch)
                loss = result["loss"]
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(batch["label"])
                count += len(batch["label"])
    if not count:
        raise ValueError("Training saw zero epochs")
    return total / count


@torch.no_grad()
def evaluate(
    model: PyHealthMaskedMLP,
    items: list[CachedRecording],
    subset: tuple[str, ...],
    batch_size: int,
    device: torch.device,
    *,
    condition: str = "",
    split: str = "",
    selected_epoch: int = 0,
    prediction_writer: csv.DictWriter | None = None,
    confusion_writer: csv.DictWriter | None = None,
    class_writer: csv.DictWriter | None = None,
) -> dict[str, float | int]:
    """Evaluate one fixed subset; optionally export predictions/diagnostics.

    Prediction identifiers are taken from the manifest (one item per recording),
    while epoch_index comes from the original indexed, resampled epoch sample.
    This preserves identity even when cached epochs are subsampled or shuffled.
    """
    model.eval()
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    loss_total, count = 0.0, 0
    subset_name = "+".join(subset)
    if any(writer is not None for writer in (prediction_writer, confusion_writer, class_writer)):
        if not condition or split not in ("val", "test") or selected_epoch < 1:
            raise ValueError("Diagnostic exports need condition, val/test split, and selected epoch")

    for item in items:
        with np.load(item.path, allow_pickle=False) as archived:
            data = {
                key: archived[key]
                for key in (*MODALITIES, "label", "epoch_index", "availability_mask")
            }
        for start in range(0, len(data["label"]), batch_size):
            ids = np.arange(start, min(start + batch_size, len(data["label"])))
            batch = to_batch(data, ids, item.participant_id, item.recording_id)
            batch = apply_modality_mask(batch, policy="eval", requested_subset=subset)
            batch = to_device(batch, device)
            result = model(batch)
            labels = result["y_true"].detach().cpu().long().numpy()
            logits = result["logit"].detach().cpu()
            if tuple(logits.shape) != (len(ids), N_CLASSES):
                raise ValueError(f"Unexpected output shape {tuple(logits.shape)}")
            predictions = logits.argmax(dim=1).numpy()
            matrix += np.bincount(
                N_CLASSES * labels + predictions,
                minlength=N_CLASSES * N_CLASSES,
            ).reshape(N_CLASSES, N_CLASSES)
            n = len(ids)
            count += n
            loss_total += float(result["loss"].item()) * n
            if prediction_writer is not None:
                probabilities = torch.softmax(logits, dim=1).numpy()
                for j, ix in enumerate(ids):
                    prediction_writer.writerow({
                        "condition": condition,
                        "split": split,
                        "subset": subset_name,
                        "selected_epoch": selected_epoch,
                        "participant_id": item.participant_id,
                        "recording_id": item.recording_id,
                        "epoch_index": int(data["epoch_index"][int(ix)]),
                        "y_true": int(labels[j]),
                        "y_pred": int(predictions[j]),
                        **{f"prob_{cls}": float(probabilities[j, cls]) for cls in range(N_CLASSES)},
                    })
    if count == 0:
        raise ValueError("No validation/test epochs")
    scores = metrics_from_confusion(matrix)
    if confusion_writer is not None:
        for true_cls in range(N_CLASSES):
            for pred_cls in range(N_CLASSES):
                confusion_writer.writerow({
                    "condition": condition,
                    "split": split,
                    "subset": subset_name,
                    "selected_epoch": selected_epoch,
                    "true_class": true_cls,
                    "true_name": CLASS_NAMES[true_cls],
                    "predicted_class": pred_cls,
                    "predicted_name": CLASS_NAMES[pred_cls],
                    "count": int(matrix[true_cls, pred_cls]),
                })
    if class_writer is not None:
        for report in class_metrics_from_confusion(matrix):
            class_writer.writerow({
                "condition": condition,
                "split": split,
                "subset": subset_name,
                "selected_epoch": selected_epoch,
                **report,
            })
    return {
        "n_epochs": count,
        "loss": loss_total / count,
        "accuracy": scores["accuracy"],
        "kappa": scores["kappa"],
    }


def score_subsets(model, items, batch_size, device, **kwargs):
    return {
        "+".join(subset): evaluate(model, items, subset, batch_size, device, **kwargs)
        for subset in SUBSETS
    }


def mean_kappa(scores) -> float:
    values = [s["kappa"] for s in scores.values() if math.isfinite(s["kappa"])]
    return float(sum(values) / len(values)) if values else float("-inf")


def run(args: argparse.Namespace) -> None:
    """Run the original ablation, exporting selected-checkpoint diagnostics."""
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch-size and lr must be positive")
    if args.max_recordings_per_split < 0 or args.max_epochs_per_recording < 0:
        raise ValueError("--max-* must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    groups = read_manifest(args.manifest, args.max_recordings_per_split)
    selected_count = sum(len(v) for v in groups.values())
    print(f"Preparing {selected_count} recordings with frozen participant splits", flush=True)
    items, stats = preprocess(groups, args.cache_dir, args.max_epochs_per_recording)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": args.seed,
        "model": "PyHealth MLP, 30-second epoch-level baseline (not wav2sleep architecture)",
        "masking": "Person 1 sample_input_mask, once per recording per training epoch",
        "evaluation": "Four fixed subsets on a common complete-case validation/test cohort",
        "checkpoint_selection": "Highest validation mean kappa across the four subsets",
        "manifest": str(args.manifest),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "max_recordings_per_split": args.max_recordings_per_split,
        "max_epochs_per_recording": args.max_epochs_per_recording,
        "sampling": "Fixed deterministic uniform random epochs per recording (not contiguous)",
        "cohort_stats": stats,
        "limitations": "SHHS1-only, nonrandom EDF cohort, subsampled epochs if capped, "
                       "no temporal context, single training seed",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    prediction_fields = [
        "condition", "split", "subset", "selected_epoch", "participant_id",
        "recording_id", "epoch_index", "y_true", "y_pred",
        *(f"prob_{cls}" for cls in range(N_CLASSES)),
    ]
    confusion_fields = [
        "condition", "split", "subset", "selected_epoch", "true_class",
        "true_name", "predicted_class", "predicted_name", "count",
    ]
    class_fields = [
        "condition", "split", "subset", "selected_epoch", "class_id",
        "class_name", "support", "predicted_count", "precision", "recall", "f1",
    ]
    subset_fields = [
        "condition", "subset", "n_epochs", "loss", "accuracy", "kappa", "selected_epoch",
    ]
    summary_fields = ["condition", "selected_epoch", "test_mean_kappa", "test_worst_kappa"]
    history_fields = ["condition", "epoch", "train_loss", "val_mean_kappa"]

    # Stream diagnostic rows to disk rather than keeping per-epoch predictions
    # in memory. Every test/validation row belongs to the selected checkpoint.
    with (
        (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as preds_f,
        (args.output_dir / "confusion_matrices.csv").open("w", newline="", encoding="utf-8") as conf_f,
        (args.output_dir / "class_metrics.csv").open("w", newline="", encoding="utf-8") as classes_f,
        (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as test_f,
        (args.output_dir / "validation_results.csv").open("w", newline="", encoding="utf-8") as val_f,
        (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as summary_f,
        (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as history_f,
    ):
        pred_writer = csv.DictWriter(preds_f, fieldnames=prediction_fields)
        conf_writer = csv.DictWriter(conf_f, fieldnames=confusion_fields)
        class_writer = csv.DictWriter(classes_f, fieldnames=class_fields)
        test_writer = csv.DictWriter(test_f, fieldnames=subset_fields)
        val_writer = csv.DictWriter(val_f, fieldnames=subset_fields)
        summary_writer = csv.DictWriter(summary_f, fieldnames=summary_fields)
        history_writer = csv.DictWriter(history_f, fieldnames=history_fields)
        for writer in (
            pred_writer, conf_writer, class_writer, test_writer,
            val_writer, summary_writer, history_writer,
        ):
            writer.writeheader()

        for condition in CONDITIONS:
            print(f"\nCONDITION: {condition}", flush=True)
            # Keep original initialization, training shuffle, mask RNG and
            # validation checkpoint-selection rule exactly the same.
            torch.manual_seed(args.seed)
            model = PyHealthMaskedMLP(hidden_dim=args.hidden_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            best_epoch, best_score = -1, float("-inf")
            best_state = None
            for epoch in range(1, args.epochs + 1):
                train_loss = train_epoch(
                    model, optimizer, items["train"], condition, epoch,
                    args.batch_size, args.seed, device,
                )
                val = score_subsets(model, items["val"], args.batch_size, device)
                score = mean_kappa(val)
                print(
                    f"  epoch={epoch} train_loss={train_loss:.4f} "
                    f"val_mean_kappa={score:.4f}", flush=True,
                )
                history_writer.writerow({
                    "condition": condition, "epoch": epoch,
                    "train_loss": train_loss, "val_mean_kappa": score,
                })
                if best_state is None or score > best_score:
                    best_epoch, best_score = epoch, score
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
            assert best_state is not None
            model.load_state_dict(best_state)
            if getattr(args, "save_checkpoints", False):
                torch.save(
                    best_state,
                    args.output_dir / f"shhs_mlp_{condition}_best.pt",
                )
            # Re-evaluate the selected checkpoint, not necessarily the final
            # training epoch, to generate aligned validation/test diagnostics.
            val_selected = score_subsets(
                model, items["val"], args.batch_size, device,
                condition=condition, split="val", selected_epoch=best_epoch,
                prediction_writer=pred_writer,
                confusion_writer=conf_writer,
                class_writer=class_writer,
            )
            test = score_subsets(
                model, items["test"], args.batch_size, device,
                condition=condition, split="test", selected_epoch=best_epoch,
                prediction_writer=pred_writer,
                confusion_writer=conf_writer,
                class_writer=class_writer,
            )
            kappas = [v["kappa"] for v in test.values() if math.isfinite(v["kappa"])]
            mean = sum(kappas) / len(kappas) if kappas else float("nan")
            worst = min(kappas) if kappas else float("nan")
            print(
                f"  best_epoch={best_epoch} test_mean_kappa={mean:.4f} "
                f"worst_kappa={worst:.4f}", flush=True,
            )
            summary_writer.writerow({
                "condition": condition,
                "selected_epoch": best_epoch,
                "test_mean_kappa": mean,
                "test_worst_kappa": worst,
            })
            for subset, result in val_selected.items():
                val_writer.writerow({
                    "condition": condition, "subset": subset, **result,
                    "selected_epoch": best_epoch,
                })
            for subset, result in test.items():
                test_writer.writerow({
                    "condition": condition, "subset": subset, **result,
                    "selected_epoch": best_epoch,
                })
            for file in (
                preds_f, conf_f, classes_f, test_f, val_f, summary_f, history_f,
            ):
                file.flush()
    print(f"\nPASS: {len(CONDITIONS)} training conditions × {len(SUBSETS)} fixed subsets = 12 evaluations")
    print(f"Results and diagnostic CSVs saved under {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/shhs_ablation"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=6250)
    parser.add_argument("--max-recordings-per-split", type=int, default=0)
    parser.add_argument("--max-epochs-per-recording", type=int, default=0)
    parser.add_argument(
        "--save-checkpoints", action="store_true",
        help="Save selected PyHealth MLP weights locally (do not commit to Git)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
