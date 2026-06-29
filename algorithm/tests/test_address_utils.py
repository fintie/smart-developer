from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algorithm.src.retrieval.hybrid_retrieve import (
    HybridRetriever,
    dedupe_results_by_address,
    minmax_norm,
    normalise_base_site_address,
)


class TestNormaliseBaseSiteAddress:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("623/21-37 WAITARA AVENUE WAITARA", "21-37 WAITARA AVENUE WAITARA"),
            ("1703/41-45 WAITARA AVENUE WAITARA", "41-45 WAITARA AVENUE WAITARA"),
            ("A12/10 GEORGE STREET SYDNEY", "10 GEORGE STREET SYDNEY"),
            ("UNIT 5 10 GEORGE STREET SYDNEY", "10 GEORGE STREET SYDNEY"),
            ("FLAT 2 14 ELIZABETH STREET", "14 ELIZABETH STREET"),
            ("  21-37   WAITARA   AVE  ", "21-37 WAITARA AVE"),
            ("21 - 37 WAITARA AVE", "21-37 WAITARA AVE"),
            ("1703 41-45 WAITARA AVENUE", "41-45 WAITARA AVENUE"),
            ("10 GEORGE STREET SYDNEY", "10 GEORGE STREET SYDNEY"),
        ],
    )
    def test_normalises_common_unit_patterns(self, raw, expected):
        assert normalise_base_site_address(raw) == expected

    def test_handles_none_and_nan(self):
        assert normalise_base_site_address(None) == ""
        assert normalise_base_site_address(float("nan")) == ""


class TestDedupeResultsByAddress:
    def test_keeps_first_per_base_site(self):
        df = pd.DataFrame(
            {
                "address": [
                    "623/21-37 WAITARA AVENUE WAITARA",
                    "1703/21-37 WAITARA AVENUE WAITARA",
                    "10 GEORGE STREET SYDNEY",
                ],
                "fusion_rank_score": [0.9, 0.7, 0.5],
            }
        )

        out = dedupe_results_by_address(df)

        assert len(out) == 2
        assert out.iloc[0]["address"] == "623/21-37 WAITARA AVENUE WAITARA"
        assert out.iloc[0]["base_site_address"] == "21-37 WAITARA AVENUE WAITARA"
        assert "dedupe_removed_count" in out.columns

    def test_noop_when_address_col_missing(self):
        df = pd.DataFrame({"score": [0.9, 0.8]})
        out = dedupe_results_by_address(df)
        assert out.equals(df)


class TestAccessPreferenceBoostSeries:
    def test_returns_zero_series_for_unknown_strategy(self):
        df = pd.DataFrame({"station_distance_band": ["within_800m", "over_10km"]})
        out = HybridRetriever._access_preference_boost_series(df, "low_rise_apartment")
        assert list(out) == [0.0, 0.0]

    def test_maps_known_strategy_bands(self):
        df = pd.DataFrame(
            {
                "station_distance_band": [
                    "within_800m",
                    "800m_2km",
                    "over_10km",
                    "unexpected_band",
                    None,
                ]
            }
        )
        out = HybridRetriever._access_preference_boost_series(
            df, "single_dwelling_rebuild"
        )
        assert list(out) == [0.05, 0.04, 0.0, 0.0, 0.0]

    def test_handles_missing_column(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        out = HybridRetriever._access_preference_boost_series(
            df, "single_dwelling_rebuild"
        )
        assert list(out) == [0.0, 0.0]


class TestMinmaxNorm:
    def test_scales_to_zero_one(self):
        s = pd.Series([1.0, 2.0, 4.0])
        out = minmax_norm(s)
        assert np.isclose(out.iloc[0], 0.0)
        assert np.isclose(out.iloc[2], 1.0)

    def test_returns_zero_when_constant(self):
        s = pd.Series([5.0, 5.0, 5.0])
        out = minmax_norm(s)
        assert (out == 0).all()

    def test_empty_series(self):
        s = pd.Series([], dtype=float)
        out = minmax_norm(s)
        assert out.empty
