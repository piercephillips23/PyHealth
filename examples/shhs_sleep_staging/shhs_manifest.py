"""Build a matched SHHS1 manifest with reproducible participant-disjoint splits.

python examples/shhs_sleep_staging/shhs_manifest.py \
    --edf-dir "<path-to-shhs1-edfs>" \
    --xml-dir "<path-to-shhs1-annotations>" \
    --output "<path-to-shhs1-manifest.csv>"

This script only inventories filenames; it does not verify EDF readability,
channel availability, annotation completeness, or signal quality.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

RECORDING_PATTERN = re.compile(r"shhs1-\d+", re.IGNORECASE)


def build_manifest(edf_dir: Path, xml_dir: Path, output: Path, seed: int = 6250) -> dict[str, int]:
    if not edf_dir.is_dir():
        raise FileNotFoundError(f"EDF directory missing: {edf_dir}")
    if not xml_dir.is_dir():
        raise FileNotFoundError(f"XML directory missing: {xml_dir}")

    edfs = {
        path.stem.lower(): path.resolve()
        for path in edf_dir.glob("*.edf")
        if RECORDING_PATTERN.fullmatch(path.stem)
    }
    xmls = {
        path.name[:-len("-nsrr.xml")].lower(): path.resolve()
        for path in xml_dir.glob("*-nsrr.xml")
        if RECORDING_PATTERN.fullmatch(path.name[:-len("-nsrr.xml")])
    }

    ids = sorted(set(edfs) & set(xmls))
    if len(ids) < 3:
        raise ValueError(f"Need at least 3 matched recordings, found {len(ids)}")

    for recording_id in ids:
        for path in (edfs[recording_id], xmls[recording_id]):
            if path.stat().st_size == 0:
                raise ValueError(f"Empty file: {path}")

    shuffled = ids.copy()
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = round(n * 0.15)
    partition = {
        recording_id: "train" if i < n_train else "val" if i < n_train + n_val else "test"
        for i, recording_id in enumerate(shuffled)
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["participant_id", "recording_id", "edf_path", "xml_path", "split"])
        writer.writeheader()
        for recording_id in ids:
            writer.writerow({
                "participant_id": recording_id,  # SHHS1-only: one visit per ID
                "recording_id": recording_id,
                "edf_path": str(edfs[recording_id]),
                "xml_path": str(xmls[recording_id]),
                "split": partition[recording_id],
            })

    counts = {
        "EDF": len(edfs),
        "XML": len(xmls),
        "paired": n,
        "EDF_missing_XML": len(set(edfs) - set(xmls)),
        "train": n_train,
        "val": n_val,
        "test": n - n_train - n_val,
    }
    print(" | ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"Saved manifest: {output.resolve()}")
    print("NOTE: Filename/size validation only; The SHHS task loader must verify channels, stages, and epoch alignment.")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf-dir", type=Path, required=True)
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6250)
    args = parser.parse_args()
    build_manifest(args.edf_dir, args.xml_dir, args.output, args.seed)


if __name__ == "__main__":
    main()
