"""Comprobaciones de los errores que alterarían la comparación temporal."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bicimad.data import TZ, deduplicate, localize_departures, make_hourly, prepare_calendar
from bicimad.features import NUMERIC, build_features
from bicimad.models import fallback_values

ROOT = Path(__file__).resolve().parents[1]


def sample_frame():
    index = pd.date_range("2022-01-01", "2023-01-01", freq="h", inclusive="left", tz=TZ).tz_convert("UTC")
    hourly = pd.DataFrame({"y": np.arange(len(index), dtype=float)}, index=index)
    calendar = pd.DataFrame({"date": pd.date_range("2022-01-01", "2022-12-31").strftime("%Y-%m-%d"), "holiday": 0})
    return hourly, calendar


def test_features_cannot_see_current_or_future_target():
    hourly, calendar = sample_frame()
    before = build_features(hourly, calendar)
    changed = hourly.copy()
    changed.iloc[300:, 0] = 1e9
    after = build_features(changed, calendar)
    pd.testing.assert_frame_equal(before[NUMERIC].iloc[:301], after[NUMERIC].iloc[:301])
    assert before.mean_24.iloc[300] == np.mean(np.arange(276, 300))


def test_missing_hour_not_zero_and_dst_not_invented():
    dates = pd.Series(["2022-01-01 00:05:00", "2022-03-27 02:20:00", "2022-10-30 02:15:00"])
    local = localize_departures(dates)
    assert local.isna().tolist() == [False, True, True]
    hourly = make_hourly(local.dropna())
    assert len(hourly) == 8760
    assert hourly.index.is_unique
    assert hourly.y.iloc[0] == 1
    assert pd.isna(hourly.y.iloc[1])
    repeated = hourly[hourly.quality.eq("ambiguous_dst")]
    assert len(repeated) == 2 and repeated.y.isna().all()


def test_deduplication_across_files_without_id():
    row = {"idBike": 1, "unlock_date": "2022-01-01T00:00:00", "lock_date": "2022-01-01T00:10:00", "station_unlock": 5, "station_lock": 6}
    seen, ids = set(), set()
    first, count, _ = deduplicate(pd.DataFrame([row, row]), seen, ids)
    assert len(first) == 1 and count == 1
    second, count, _ = deduplicate(pd.DataFrame([row]), seen, ids)
    assert second.empty and count == 1


def test_id_deduplication_even_with_changed_metadata():
    rows = pd.DataFrame({"idTrip": ["synthetic_A", "synthetic_A"], "idBike": [1, 1],
                         "unlock_date": ["2022-01-01T00:00:00"] * 2,
                         "lock_date": ["2022-01-01T00:10:00"] * 2,
                         "station_unlock": [5, 5], "station_lock": [6, 7]})
    clean, count, id_count = deduplicate(rows, set(), set())
    assert len(clean) == 1 and count == id_count == 1


def test_split_and_warmup():
    hourly, calendar = sample_frame()
    frame = build_features(hourly, calendar)
    assert frame.groupby("split").size().to_dict() == {"test": 1464, "train": 5831, "validation": 1465}
    assert frame.iloc[:168].eligible.sum() == 0
    assert frame.loc[frame.split.eq("validation"), "lag_1"].iloc[0] == frame.loc[frame.split.eq("train"), "y"].iloc[-1]


def test_unknown_calendar_and_irregular_grid_fail():
    hourly, calendar = sample_frame()
    with pytest.raises(ValueError, match="calendario"):
        build_features(hourly, calendar.iloc[1:])
    with pytest.raises(ValueError, match="rejilla"):
        build_features(hourly.drop(hourly.index[10]), calendar)


def test_scaler_and_imputer_only_fit_training():
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    pipeline.fit(pd.DataFrame({"x": [1., 3., np.nan]}))
    pipeline.transform(pd.DataFrame({"x": [1e9, np.nan]}))
    assert pipeline[0].statistics_[0] == 2
    assert pipeline[1].mean_[0] == 2


def test_fallback_uses_training_values():
    train = pd.DataFrame({"weekday": [0, 0, 1], "hour": [8, 8, 9], "y": [10., 30., 50.]})
    valid = pd.DataFrame({"weekday": [0, 6], "hour": [8, 1], "y": [1e9, 1e9]})
    assert fallback_values(train, valid).tolist() == [20., 30.]


def test_official_calendar_covers_year():
    path = ROOT / "data/processed/calendar_2022.csv"
    if not path.exists():
        pytest.skip("Se genera con prepare.")
    calendar = pd.read_csv(path)
    assert len(calendar) == 365 and calendar.date.is_unique
    assert calendar.set_index("date").loc["2022-05-16", "holiday"] == 1


def test_saved_predictions_exclude_holdout():
    path = ROOT / "results/predictions_validation.csv"
    if not path.exists():
        pytest.skip("Se genera con train.")
    predictions = pd.read_csv(path)
    times = pd.to_datetime(predictions.time_utc, utc=True).dt.tz_convert(TZ)
    assert times.dt.month.isin([9, 10]).all()
    assert not predictions.isna().any().any()
