"""
Ragam definitions and loader.

Reads ragams/melakartas.yaml, validates the schema, converts swara symbols
to cents values, and exposes ragams as queryable dataclasses.

Validation runs at load time so malformed YAML or domain-rule violations
(sampurna structure, swara consistency) fail loudly with the offending entry
identified, rather than producing silent wrong-answer bugs downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

import yaml

from src.swara.definitions import (
    VALID_SWARA_TOKENS,
    swara_to_cents,
    swaras_to_cents,
)

# Default location of the ragam YAML, relative to project root.
DEFAULT_RAGAMS_PATH = Path("ragams") / "melakartas.yaml"

# Expected total number of melakarta ragams.
EXPECTED_MELAKARTA_COUNT = 72

# Sampurna melakartas have exactly 8 swaras in arohanam (7 unique + upper S).
EXPECTED_AROHANAM_LENGTH = 8


@dataclass(frozen=True)
class Ragam:
    """
    A Carnatic ragam definition with pre-computed cents values.

    Frozen because ragam definitions are canonical references — downstream
    code reads against them; nothing mutates them at runtime.

    Fields:
        number: melakarta number (1-72)
        name: common name
        arohanam / avarohanam: swara symbol sequences
        swaras: unique swara set used by the ragam
        arohanam_cents / avarohanam_cents: cents values aligned with symbol lists
        swaras_cents: cents values as a frozenset for O(1) membership checks
        allied_ragas: ragas commonly confused with this one
        melakarta_name: formal Sanskrit name (only when it differs from `name`)
        notes: optional context
    """
    number: int
    name: str
    arohanam: List[str]
    avarohanam: List[str]
    swaras: List[str]
    arohanam_cents: List[float]
    avarohanam_cents: List[float]
    swaras_cents: FrozenSet[float]
    allied_ragas: List[str] = field(default_factory=list)
    melakarta_name: Optional[str] = None
    notes: Optional[str] = None

    def contains_swara(self, cents: float, tolerance: float = 0.0) -> bool:
        """
        Check if a cents value matches any swara in this ragam.

        Used by the adherence checker. Caller is expected to pass cents
        already octave-folded to [0, 1200).

        Args:
            cents: pitch in cents, octave-reduced.
            tolerance: cents window for fuzzy matching (0 = exact match).

        Returns:
            True if `cents` is within `tolerance` of any allowed swara.
        """
        if tolerance == 0.0:
            return cents in self.swaras_cents
        return any(
            abs(cents - allowed) <= tolerance
            for allowed in self.swaras_cents
        )


class RagamLoadError(ValueError):
    """Raised when ragam YAML fails to parse or violates schema/domain rules."""
    pass


class RagamRegistry:
    """
    In-memory collection of all loaded ragams.

    Supports lookup by melakarta number, common name, and formal melakarta name.
    Built once from YAML; instances are immutable in practice (no setters exposed).
    """

    def __init__(self, ragams: List[Ragam]) -> None:
        self._by_number: Dict[int, Ragam] = {r.number: r for r in ragams}
        self._by_name: Dict[str, Ragam] = {r.name.lower(): r for r in ragams}
        # Also index by formal melakarta_name where it differs from common name.
        for r in ragams:
            if r.melakarta_name and r.melakarta_name.lower() not in self._by_name:
                self._by_name[r.melakarta_name.lower()] = r

    def get_by_number(self, number: int) -> Ragam:
        """Look up a ragam by its melakarta number (1-72)."""
        if number not in self._by_number:
            raise KeyError(f"No ragam with melakarta number {number}")
        return self._by_number[number]

    def get_by_name(self, name: str) -> Ragam:
        """Case-insensitive lookup by common name or formal melakarta name."""
        key = name.lower()
        if key not in self._by_name:
            raise KeyError(f"No ragam with name {name!r}")
        return self._by_name[key]

    def all_ragams(self) -> List[Ragam]:
        """Return all loaded ragams, ordered by melakarta number."""
        return sorted(self._by_number.values(), key=lambda r: r.number)

    def __len__(self) -> int:
        return len(self._by_number)


def _validate_ragam_entry(entry: dict, idx: int) -> None:
    """
    Validate a single ragam YAML entry. Raises RagamLoadError with a specific
    message identifying the offending entry on failure.
    """
    required = {"number", "name", "arohanam", "avarohanam", "swaras", "allied_ragas"}
    missing = required - set(entry.keys())
    if missing:
        raise RagamLoadError(
            f"Ragam entry #{idx} (number={entry.get('number')}) "
            f"missing required fields: {sorted(missing)}"
        )

    # All swara tokens must be recognized swarasthanas.
    for field_name in ("arohanam", "avarohanam", "swaras"):
        for token in entry[field_name]:
            if token not in VALID_SWARA_TOKENS:
                raise RagamLoadError(
                    f"Ragam #{entry['number']} ({entry['name']}): "
                    f"unknown swara token {token!r} in {field_name}."
                )

    # Sampurna structure: 8 swaras in arohanam (S ... S').
    if len(entry["arohanam"]) != EXPECTED_AROHANAM_LENGTH:
        raise RagamLoadError(
            f"Ragam #{entry['number']} ({entry['name']}): arohanam must have "
            f"exactly {EXPECTED_AROHANAM_LENGTH} swaras; got {len(entry['arohanam'])}"
        )

    # Arohanam starts with S, ends with S'. Avarohanam mirrors.
    if entry["arohanam"][0] != "S" or entry["arohanam"][-1] != "S'":
        raise RagamLoadError(
            f"Ragam #{entry['number']} ({entry['name']}): arohanam must start "
            f"with S and end with S'"
        )
    if entry["avarohanam"][0] != "S'" or entry["avarohanam"][-1] != "S":
        raise RagamLoadError(
            f"Ragam #{entry['number']} ({entry['name']}): avarohanam must start "
            f"with S' and end with S"
        )

    # `swaras` field must match the unique non-upper-Sa swaras in arohanam.
    arohanam_set = {s for s in entry["arohanam"] if s != "S'"}
    swaras_set = set(entry["swaras"])
    if arohanam_set != swaras_set:
        raise RagamLoadError(
            f"Ragam #{entry['number']} ({entry['name']}): `swaras` field "
            f"{sorted(swaras_set)} doesn't match unique swaras in arohanam "
            f"{sorted(arohanam_set)}"
        )


def load_ragams(path: Optional[Path] = None) -> RagamRegistry:
    """
    Load and validate all ragams from YAML.

    Args:
        path: path to the YAML file. Defaults to ragams/melakartas.yaml
            relative to the current working directory.

    Returns:
        A RagamRegistry containing all parsed ragams.

    Raises:
        FileNotFoundError: if the YAML file doesn't exist.
        RagamLoadError: if YAML is malformed or any entry fails validation.
    """
    yaml_path = path if path is not None else DEFAULT_RAGAMS_PATH

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Ragams YAML not found at {yaml_path.resolve()}. "
            f"Pass `path=` explicitly if running from outside the project root."
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "ragams" not in data:
        raise RagamLoadError(
            f"Top-level YAML must be a dict with a `ragams:` key; "
            f"got {type(data).__name__}"
        )

    entries = data["ragams"]
    if not isinstance(entries, list):
        raise RagamLoadError(
            f"`ragams:` must be a list; got {type(entries).__name__}"
        )

    ragams: List[Ragam] = []
    seen_numbers: set[int] = set()

    for idx, entry in enumerate(entries):
        _validate_ragam_entry(entry, idx)

        if entry["number"] in seen_numbers:
            raise RagamLoadError(
                f"Duplicate melakarta number {entry['number']} at entry #{idx}"
            )
        seen_numbers.add(entry["number"])

        ragam = Ragam(
            number=entry["number"],
            name=entry["name"],
            arohanam=list(entry["arohanam"]),
            avarohanam=list(entry["avarohanam"]),
            swaras=list(entry["swaras"]),
            arohanam_cents=swaras_to_cents(entry["arohanam"]),
            avarohanam_cents=swaras_to_cents(entry["avarohanam"]),
            swaras_cents=frozenset(swaras_to_cents(entry["swaras"])),
            allied_ragas=list(entry.get("allied_ragas", [])),
            melakarta_name=entry.get("melakarta_name"),
            notes=entry.get("notes"),
        )
        ragams.append(ragam)

    if len(ragams) != EXPECTED_MELAKARTA_COUNT:
        raise RagamLoadError(
            f"Expected {EXPECTED_MELAKARTA_COUNT} melakarta ragams, "
            f"got {len(ragams)}"
        )

    return RagamRegistry(ragams)