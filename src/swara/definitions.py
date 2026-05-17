"""
Swarasthana definitions for Carnatic music.

Defines the 12-tone equal temperament (12-TET) cents values for each
swarasthana relative to Sa (S = 0 cents). All downstream pipeline components
(segmenter, adherence checker, identifier features) consume cents, not Hz.

Notes on the 12-TET choice:
    Carnatic theory traditionally uses just-intonation ratios (e.g. G3 at
    ratio 5/4 = ~386 cents rather than 400). For the MVP we use 12-TET as
    the baseline because:
      1. The downstream cents tolerance (config.yaml `swara.tolerance_cents`,
         default 35) is wider than the max 12-TET-vs-JI gap (~14c at G3/N2),
         so the choice is absorbed by tolerance.
      2. Most pitch libraries report 12-TET-aligned frequencies natively.
      3. Swapping to JI later is a single-file change here; no callers need
         to update.

Upper-octave Sa is represented as 'S\\'' (parsed from YAML as 'S\\'') and
maps to 1200 cents. This module handles octave wrap-around so callers can
write swara symbols directly without managing octaves themselves.
"""

from __future__ import annotations

from typing import Dict, List

# Cents value of each swarasthana relative to Sa (S = 0).
# Note that some positions share cents values across different swara names
# (e.g. R2 and G1 both sit at 200 cents). This is consistent with Carnatic
# theory: the same pitch position can function as either swara depending on
# which swaras are present in the raga. Ragam definitions disambiguate.
SWARASTHANA_CENTS: Dict[str, float] = {
    "S":  0.0,
    "R1": 100.0,
    "R2": 200.0,
    "R3": 300.0,
    "G1": 200.0,
    "G2": 300.0,
    "G3": 400.0,
    "M1": 500.0,
    "M2": 600.0,
    "P":  700.0,
    "D1": 800.0,
    "D2": 900.0,
    "D3": 1000.0,
    "N1": 900.0,
    "N2": 1000.0,
    "N3": 1100.0,
}

# Octave marker for upper Sa as it appears in ragam YAML files.
UPPER_OCTAVE_SUFFIX = "'"

# All valid swara tokens (used by the YAML loader for validation).
VALID_SWARA_TOKENS: frozenset[str] = frozenset(
    list(SWARASTHANA_CENTS.keys()) + [f"S{UPPER_OCTAVE_SUFFIX}"]
)


def swara_to_cents(swara: str) -> float:
    """
    Convert a swara symbol to its cents value relative to Sa.

    Handles upper-octave Sa (S') by adding 1200 cents.

    Args:
        swara: Swara symbol (e.g. 'S', 'R1', 'G3', "S'").

    Returns:
        Cents offset from Sa.

    Raises:
        ValueError: If `swara` is not a recognized swarasthana.
    """
    if swara == f"S{UPPER_OCTAVE_SUFFIX}":
        return SWARASTHANA_CENTS["S"] + 1200.0

    if swara not in SWARASTHANA_CENTS:
        raise ValueError(
            f"Unknown swara symbol: {swara!r}. "
            f"Valid tokens: {sorted(VALID_SWARA_TOKENS)}"
        )

    return SWARASTHANA_CENTS[swara]


def swaras_to_cents(swaras: List[str]) -> List[float]:
    """
    Convert a sequence of swara symbols to cents values.

    Used by the ragam loader to translate `arohanam` / `avarohanam` /
    `swaras` lists from YAML into numeric pitch sets the segmenter can
    match pitch frames against.

    Args:
        swaras: List of swara symbols.

    Returns:
        List of cents values in the same order as `swaras`.
    """
    return [swara_to_cents(s) for s in swaras]


def cents_mod_octave(cents: float) -> float:
    """
    Reduce a cents value to its position within a single octave [0, 1200).

    The segmenter operates on octave-folded cents so that the same swara
    sung at different octaves matches the same ragam definition entry.

    Args:
        cents: Cents value (can be negative or >= 1200).

    Returns:
        Cents value in the half-open interval [0, 1200).
    """
    return cents % 1200.0