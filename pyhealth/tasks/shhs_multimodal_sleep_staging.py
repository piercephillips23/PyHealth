"""Mask-aware SHHS epoch task for reproducing wav2sleep modality ablations."""

import logging
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pyhealth.tasks.base_task import BaseTask
from pyhealth.tasks.sleep_staging_nsrr import (
    CHANNEL_ALIASES,
    EPOCH_SECONDS,
    LABEL_SCHEMES,
    POINTS_PER_EPOCH,
    _read_edf_channels,
    _read_stage_codes,
    _to_epochs,
)

logger = logging.getLogger(__name__)
MODALITIES = ("ecg", "abd", "thx")


class SHHSMultimodalSleepStaging(BaseTask):
    """Build deterministic, mask-aware samples from SHHS recordings.

    Reuses PyHealth's native-rate NSRR reader and anti-aliased full-recording
    resampling. The task adds fixed-shape placeholders, natural availability,
    fixed four-class labels, and metadata for participant-level splitting.
    Missing, non-finite, or near-constant channels are marked unavailable.
    Normalization uses the aligned recording (offline, not causal). Unknown
    labels are skipped, as well as recordings with no usable selected channel are skipped.

    Args:
        modalities: Non-empty subset of ECG, ABD and THX. All three output 
            fields remain present regardless of this selection.
        normalize: Apply per-recording/channel z-score normalization before
            polyphase resampling.
        min_std: Minimum standard deviation in EDF physical units for a
            channel to count as usable.
    """

    task_name = "SHHSMultimodalSleepStaging"

    def __init__(
        self,
        modalities: Sequence[str] = MODALITIES,
        normalize: bool = True,
        min_std: float = 1e-6,
    ) -> None:
        super().__init__()

        selected = tuple(modalities)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("modalities must be a nonempty subset without duplicates")
        if any(m not in MODALITIES for m in selected):
            raise ValueError(f"modalities must be drawn from {modalities}")
        if not np.isfinite(min_std) or min_std <= 0:
            raise ValueError("min_std must be positive and finite")
        
        self.modalities = selected
        self.normalize = normalize
        self.min_std = min_std
        self.input_schema: dict[str, Any] = {m: "tensor" for m in modalities}
        self.input_schema["availability_mask"] = ("tensor", {"dtype": torch.bool})
        # A fixed vocabulary preserves class indices even if a tiny cohort
        # does not contain every sleep stage.
        self.output_schema: dict[str, Any] = {"label": ("multiclass", {"labels": 4})}

    def load_recording(
        self, edf_path: str | Path, xml_path: str | Path, patient_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Load one EDF/XML pair; raise clear errors for malformed input.

        Args:
            edf_path: SHHS1 EDF file.
            xml_path: Matching NSRR or profusion XML file.
            patient_id: Optional dataset patient identifier; defaults to the
                SHHS1 numeric ID in the EDF filename.

        Returns:
            One dictionary per scored epoch, retaining the original index.
            Signals are float32 arrays; the availability mask is bool[3].
        """
        recording_id = Path(edf_path).stem
        stages = _read_stage_codes(str(xml_path))
        aliases = [a for m in MODALITIES for a in CHANNEL_ALIASES[m]]
        channels = _read_edf_channels(str(edf_path), aliases)
        present: dict[str, tuple[np.ndarray, float]] = {}

        for m in MODALITIES:
            if (m not in self.modalities):
                continue
            hit = next((channels[a] for a in CHANNEL_ALIASES[m] if a in channels), None)
            if (hit is None):
                continue
            signal, rate = hit
            count = min(len(signal), round(len(stages) * EPOCH_SECONDS * rate))
            usable = signal[:count]
            if ((count < round(EPOCH_SECONDS * rate)) or (not np.isfinite(usable).all()) or (np.std(usable, dtype=np.float64) < self.min_std)):
                continue
            present[m] = (signal, rate)

        if not present:
            raise ValueError("No usable selected physiological channel")
        
        n_epochs = min(
            len(stages),
            *(int(len(signal) // (rate * EPOCH_SECONDS))
              for signal, rate in present.values()),
        )
        if n_epochs != len(stages):
            logger.warning(
                "Discarding trailing stage labels without full signal epochs"
            )
        if n_epochs == 0:
            raise ValueError("No complete aligned epochs")
        
        waveforms = {
            m: _to_epochs(signal, rate, n_epochs, POINTS_PER_EPOCH[m], self.normalize)
            for m, (signal, rate) in present.items()
        }
        # Check after resampling too; never send nonfinite values to a model.
        waveforms = {m: x for m, x in waveforms.items() if np.isfinite(x).all()}
        if not waveforms:
            raise ValueError("No finite resampled channels")
        
        availability = np.asarray([m in waveforms for m in MODALITIES], dtype=bool)
        zeros = {m: np.zeros(POINTS_PER_EPOCH[m], np.float32) for m in MODALITIES}
        labels = LABEL_SCHEMES["four_class"]
        samples = []
        for i, code in enumerate(stages[:n_epochs]):
            if int(code) not in labels:
                continue

            sample = {
                m: waveforms[m][i].copy() if m in waveforms else zeros[m].copy()
                for m in MODALITIES
            }
            sample.update(
                patient_id=patient_id if patient_id is not None else recording_id[6:],
                recording_id=recording_id,
                epoch_index=i,
                availability_mask=availability.copy(),
                label=labels[int(code)],
            )

            samples.append(sample)

        return samples

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Extract valid SHHS epochs from an NSRRSleepDataset patient.

        Args:
            patient: PyHealth patient exposing EDF/XML events.

        Returns:
            Deterministic samples, with unreadable recordings logged and skipped.
        """
        samples = []
        for event in patient.get_events():
            samples.extend(self.load_recording(
                event.signal_file, event.label_file, patient.patient_id
            ))

        return samples
