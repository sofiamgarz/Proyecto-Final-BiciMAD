"""Variables disponibles justo al inicio de la hora que se quiere predecir."""

import numpy as np
import pandas as pd

from .data import TZ

HISTORY = ["lag_1", "lag_24", "lag_168", "mean_24"]
NUMERIC = HISTORY + ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos", "holiday", "weekend"]


def read_hourly(path):
    frame = pd.read_csv(path)
    frame.index = pd.to_datetime(frame.pop("time_utc"), utc=True)
    return frame


def build_features(hourly, calendar):
    if not hourly.index.is_monotonic_increasing or not hourly.index.is_unique:
        raise ValueError("La serie debe estar ordenada y no repetir instantes UTC.")
    if not (hourly.index[1:] - hourly.index[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("Se necesita una rejilla regular antes de calcular retardos.")
    frame = hourly.copy()
    local = frame.index.tz_convert(TZ)
    frame["hour"], frame["weekday"], frame["month"] = local.hour, local.day_of_week, local.month
    for name, period in [("hour", 24), ("weekday", 7), ("month", 12)]:
        frame[name + "_sin"] = np.sin(2 * np.pi * frame[name] / period)
        frame[name + "_cos"] = np.cos(2 * np.pi * frame[name] / period)
    holidays = calendar.set_index("date")["holiday"]
    values = pd.Series(local.strftime("%Y-%m-%d")).map(holidays)
    if values.isna().any():
        raise ValueError("Faltan fechas en el calendario.")
    frame["holiday"] = values.to_numpy()
    frame["weekend"] = (local.day_of_week >= 5).astype(int)
    frame["day_type"] = np.select([frame.holiday.eq(1), frame.weekend.eq(1)], ["festivo", "fin_semana"], default="laborable")
    for hours in [1, 24, 168]:
        frame[f"lag_{hours}"] = frame.y.shift(hours)
    frame["mean_24"] = frame.y.shift(1).rolling(24, min_periods=24).mean()
    frame["split"] = np.select([local.month <= 8, local.month <= 10], ["train", "validation"], default="test")
    frame["warmup"] = np.arange(len(frame)) < 168
    frame["eligible"] = frame.y.notna() & ~frame.warmup
    return frame
