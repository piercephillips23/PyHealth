"""Dynamic masking for fixed-shape SHHS samples; not part of cached tasks."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch

from pyhealth.tasks.shhs_multimodal_sleep_staging import MODALITIES
from pyhealth.tasks.sleep_staging_nsrr import POINTS_PER_EPOCH

MaskPolicy = Literal["none", "paper", "balanced", "eval"]
PAPER_DROP = np.array([0.5, 0.7, 0.7])


def sample_input_mask(
    availability: np.ndarray | torch.Tensor | Sequence[bool],
    policy: MaskPolicy,
    rng: np.random.Generator | None = None,
    requested_subset: Sequence[str] | None = None,
) -> np.ndarray:
    """Choose a nonempty subset of available channels in ECG/ABD/THX order.

    Args:
        availability: Boolean vector of length three.
        policy: ``none``, ``paper`` (independent drops conditioned on at least
            one retained channel), ``balanced`` (uniform nonempty subset),
            or ``eval`` (fixed intersection with requested channels).
        rng: Persistent NumPy generator, required for stochastic policies.
        requested_subset: Lowercase channel names for ``eval`` only.

    Returns:
        Boolean vector of length three. Never mutates availability.

    Raises:
        ValueError: On invalid policy, subset, mask, or no available input.
    """
    if isinstance(availability, torch.Tensor):
        availability = availability.detach().cpu().numpy()
    available = np.asarray(availability)
    if available.shape != (3,) or available.dtype != np.bool_ or not available.any():
        raise ValueError("availability must be a nonempty boolean vector of length 3")
    if requested_subset is not None and policy != "eval":
        raise ValueError("requested_subset is only valid for eval")
    if policy == "none":
        return available.copy()
    if policy == "eval":
        if not requested_subset or any(m not in MODALITIES for m in requested_subset):
            raise ValueError(f"eval requires a nonempty subset of {MODALITIES}")
        mask = available & np.array([m in requested_subset for m in MODALITIES])
        if not mask.any():
            raise ValueError("Requested evaluation subset has no available channel")
        return mask
    if policy not in ("paper", "balanced") or rng is None:
        raise ValueError(
            "Stochastic policies require a valid policy and persistent RNG"
        )
    if policy == "paper":
        while True:
            mask = available & (rng.random(3) >= PAPER_DROP)
            if mask.any():
                return mask
    positions = np.flatnonzero(available)
    bits = int(rng.integers(1, 2 ** len(positions)))
    mask = np.zeros(3, dtype=bool)
    for i, position in enumerate(positions):
        mask[position] = bool(bits & (1 << i))
    return mask


def collate_shhs_epochs(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack raw or processed epoch samples while preserving split metadata.

    Args:
        samples: Nonempty sequence of task samples.

    Returns:
        Model-ready tensor batch with natural availability and integer labels.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    batch = {
        m: torch.stack([torch.as_tensor(s[m], dtype=torch.float32) for s in samples])
        for m in MODALITIES
    }
    batch["availability_mask"] = torch.stack([
        torch.as_tensor(s["availability_mask"], dtype=torch.bool) for s in samples
    ])
    batch["label"] = torch.stack([
        torch.as_tensor(s["label"], dtype=torch.int64) for s in samples
    ])
    for key in ("patient_id", "recording_id", "epoch_index"):
        batch[key] = [s[key] for s in samples]
    return batch


def apply_modality_mask(
    batch: Mapping[str, Any],
    policy: MaskPolicy,
    rng: np.random.Generator | None = None,
    requested_subset: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a new batch with zeroed waveforms and explicit input_mask.

    Draws are per epoch, not per night as in the paper. Call anew each
    training pass with the same advancing RNG; do not cache the result.
    Evaluation is deterministic and fails if any row has no requested input.

    Args:
        batch: Collated SHHS samples (CPU or GPU waveform tensors).
        policy: Masking policy passed to :func:`sample_input_mask`.
        rng: Persistent generator for stochastic policies.
        requested_subset: Fixed test-time channels for the evaluation policy.

    Returns:
        Copy of the batch dictionary with masked waveform tensors and a
        bool[B,3] input_mask on the availability mask's device. Natural
        availability and labels are not modified.
    """
    available = batch["availability_mask"]
    if not isinstance(available, torch.Tensor) or available.dtype != torch.bool:
        raise ValueError("availability_mask must be a boolean tensor")
    if available.ndim != 2 or available.shape[1] != 3 or len(available) == 0:
        raise ValueError("availability_mask must have shape [B,3], B > 0")
    rows = available.detach().cpu().numpy()
    mask = torch.as_tensor(np.stack([
        sample_input_mask(row, policy, rng, requested_subset) for row in rows
    ]), device=available.device)
    output = dict(batch)
    output["input_mask"] = mask
    for i, m in enumerate(MODALITIES):
        waveform = batch[m]
        if waveform.shape != (len(available), POINTS_PER_EPOCH[m]):
            raise ValueError(f"Unexpected shape for {m}: {waveform.shape}")
        output[m] = waveform.masked_fill(~mask[:, i, None].to(waveform.device), 0)
    return output
