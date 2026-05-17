"""Tests for the ragam loader and registry."""

import pytest

from src.swara.definitions import swara_to_cents
from src.swara.ragams import (
    EXPECTED_MELAKARTA_COUNT,
    Ragam,
    RagamRegistry,
    load_ragams,
)


@pytest.fixture(scope="module")
def registry() -> RagamRegistry:
    """Load the full 72-ragam registry once per test module."""
    return load_ragams()


class TestLoading:

    def test_loads_all_72_melakartas(self, registry):
        assert len(registry) == EXPECTED_MELAKARTA_COUNT

    def test_all_numbers_1_to_72_present(self, registry):
        numbers = {r.number for r in registry.all_ragams()}
        assert numbers == set(range(1, 73))

    def test_all_entries_are_ragam_instances(self, registry):
        for ragam in registry.all_ragams():
            assert isinstance(ragam, Ragam)


class TestLookup:

    def test_lookup_by_number(self, registry):
        sankarabharanam = registry.get_by_number(29)
        assert sankarabharanam.name == "Sankarabharanam"

    def test_lookup_by_common_name(self, registry):
        kalyani = registry.get_by_name("Kalyani")
        assert kalyani.number == 65

    def test_lookup_by_name_is_case_insensitive(self, registry):
        kalyani = registry.get_by_name("kalyani")
        assert kalyani.number == 65

    def test_lookup_by_melakarta_name(self, registry):
        kalyani = registry.get_by_name("Mechakalyani")
        assert kalyani.number == 65

    def test_invalid_number_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get_by_number(100)

    def test_invalid_name_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get_by_name("not-a-ragam")


class TestSwaraContent:

    def test_sankarabharanam_swara_set_correct(self, registry):
        """Sankarabharanam (#29) uses S R2 G3 M1 P D2 N3."""
        ragam = registry.get_by_number(29)
        expected = frozenset({
            swara_to_cents("S"),
            swara_to_cents("R2"),
            swara_to_cents("G3"),
            swara_to_cents("M1"),
            swara_to_cents("P"),
            swara_to_cents("D2"),
            swara_to_cents("N3"),
        })
        assert ragam.swaras_cents == expected

    def test_contains_swara_exact(self, registry):
        ragam = registry.get_by_number(29)
        assert ragam.contains_swara(swara_to_cents("G3"))

    def test_contains_swara_with_tolerance(self, registry):
        """G3 = 400 cents in 12-TET; 395c should match with ±10 tolerance."""
        ragam = registry.get_by_number(29)
        assert ragam.contains_swara(395.0, tolerance=10.0)

    def test_does_not_contain_foreign_swara(self, registry):
        """Sankarabharanam uses R2 (200), not R1 (100)."""
        ragam = registry.get_by_number(29)
        assert not ragam.contains_swara(swara_to_cents("R1"))

    def test_swaras_cents_excludes_upper_octave(self, registry):
        """The adherence-check swara set lives in [0, 1200); no upper Sa."""
        for ragam in registry.all_ragams():
            assert 1200.0 not in ragam.swaras_cents