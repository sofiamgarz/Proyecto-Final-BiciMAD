"""Descarga verificada y limpieza de los doce CSV oficiales, sin publicar viajes."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TZ = "Europe/Madrid"
KEY = ["idBike", "unlock_date", "lock_date", "station_unlock", "station_lock"]
REQUIRED = KEY + ["fleet", "unlocktype", "trip_minutes"]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(root):
    """Descarga sin extraer rutas de los ZIP y exige la versión inventariada."""
    sources = json.loads((root / "config/sources.json").read_text())
    for source in sources:
        path = root / "data/raw" / source["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            with requests.get(source["url"], stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            temporary.replace(path)
        if sha256(path) != source["sha256"]:
            raise ValueError(f"La fuente ha cambiado: {path.name}. Revisar antes de continuar.")
        print(f"Verificado: {source['path']}", flush=True)


def localize_departures(values):
    """Hipótesis de hora local; no inventa el desplazamiento de la hora repetida."""
    naive = pd.to_datetime(values, errors="coerce")
    return naive.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")


def deduplicate(frame, seen_keys, seen_ids):
    """Primera aparición estable; hash de clave compuesta y de idTrip si existe.

    No se escribe ningún identificador individual en los productos públicos.
    La clave compuesta permite comparar también los meses sin idTrip.
    """
    keys = pd.util.hash_pandas_object(frame[KEY].astype("string"), index=False)
    duplicate = keys.duplicated() | keys.isin(seen_keys)
    seen_keys.update(keys.tolist())
    id_duplicate = pd.Series(False, index=frame.index)
    if "idTrip" in frame:
        valid = frame.idTrip.notna()
        ids = pd.util.hash_pandas_object(frame.loc[valid, "idTrip"].astype("string"), index=False)
        id_duplicate.loc[valid] = ids.duplicated() | ids.isin(seen_ids)
        seen_ids.update(ids.tolist())
    combined = duplicate | id_duplicate
    return frame.loc[~combined].copy(), int(combined.sum()), int(id_duplicate.sum())


def make_hourly(times):
    """Rejilla UTC regular; una hora sin registros queda desconocida, nunca cero."""
    grid = pd.date_range("2022-01-01", "2023-01-01", freq="h", inclusive="left", tz=TZ)
    grid = grid.tz_convert("UTC")
    counts = times.dt.tz_convert("UTC").dt.floor("h").value_counts().reindex(grid)
    hourly = pd.DataFrame({"y": counts.astype(float)}, index=grid)
    hourly.index.name = "time_utc"
    local = hourly.index.tz_convert(TZ)
    hourly["time_local"] = local.astype(str)
    ambiguous = local.tz_localize(None).duplicated(keep=False)
    hourly["quality"] = np.where(hourly.y.notna(), "observed", "no_records_unknown")
    hourly.loc[ambiguous, "y"] = np.nan
    hourly.loc[ambiguous, "quality"] = "ambiguous_dst"
    return hourly


def preprocess(root):
    """Limpia todo el año con reglas fijas; no usa estadísticas de prueba."""
    raw = root / "data/raw"
    audit, times, distributions = [], [], []
    seen_keys, seen_ids = set(), set()
    with zipfile.ZipFile(raw / "bicimad_2022.zip") as annual:
        members = sorted(n for n in annual.namelist() if n.startswith("bicimad_2022/trips_") and n.endswith(".zip"))
        if len(members) != 12:
            raise ValueError("Se requieren exactamente doce archivos mensuales de viajes.")
        for member in members:
            with zipfile.ZipFile(io.BytesIO(annual.read(member))) as monthly:
                csvs = [n for n in monthly.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX/")]
                if len(csvs) != 1:
                    raise ValueError(f"Contenido mensual inesperado: {member}")
                with monthly.open(csvs[0]) as stream:
                    frame = pd.read_csv(stream, sep=";", low_memory=False)
            missing = set(REQUIRED) - set(frame.columns)
            if missing:
                raise ValueError(f"Columnas ausentes: {missing}")
            month = int(Path(member).name.split("_")[2])
            a = {"month": month, "rows_read": len(frame), "columns": len(frame.columns), "has_idTrip": "idTrip" in frame}
            empty = frame.isna().all(axis=1)
            a["empty_rows"] = int(empty.sum())
            frame = frame.loc[~empty].copy()
            a["nonempty_rows"] = len(frame)
            a["missing_unlock_date"] = int(frame.unlock_date.isna().sum())
            a["missing_idBike"] = int(frame.idBike.isna().sum())
            a["missing_station_unlock"] = int(frame.station_unlock.isna().sum())
            frame, a["duplicates_removed"], a["duplicate_idTrip"] = deduplicate(frame, seen_keys, seen_ids)
            for (fleet, kind), n in frame.groupby(["fleet", "unlocktype"], dropna=False).size().items():
                distributions.append({"month": month, "fleet": fleet, "unlocktype": kind, "n": int(n)})
            station = pd.to_numeric(frame.station_unlock, errors="coerce")
            mask = (frame.fleet == 1) & frame.unlocktype.eq("STATION") & station.gt(0) & station.mod(1).eq(0)
            a["outside_scope"] = int((~mask).sum())
            frame = frame.loc[mask].copy()
            naive = pd.to_datetime(frame.unlock_date, errors="coerce")
            local = localize_departures(frame.unlock_date)
            a["invalid_datetime"] = int(naive.isna().sum())
            a["dst_unresolved"] = int((naive.notna() & local.isna()).sum())
            in_month = local.notna() & local.dt.year.eq(2022) & local.dt.month.eq(month)
            a["outside_month"] = int((local.notna() & ~in_month).sum())
            frame, local = frame.loc[in_month], local.loc[in_month]
            duration = pd.to_numeric(frame.trip_minutes, errors="coerce")
            a["duration_missing"] = int(duration.isna().sum())
            a["duration_nonpositive"] = int(duration.le(0).sum())
            a["duration_over_24h"] = int(duration.gt(1440).sum())
            a["missing_station_lock"] = int(frame.station_lock.isna().sum())
            # Se cuentan salidas: duración/destino son controles, no filtros futuros.
            a["departures_counted"] = len(frame)
            times.append(local)
            audit.append(a)
            print(f"Mes {month:02}: {len(frame):,} salidas; {a['duplicates_removed']:,} duplicados", flush=True)
    hourly = make_hourly(pd.concat(times, ignore_index=True))
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    results = root / "results"
    results.mkdir(exist_ok=True)
    hourly.to_csv(processed / "hourly_2022.csv")
    pd.DataFrame(audit).to_csv(results / "quality_monthly.csv", index=False)
    pd.DataFrame(distributions).to_csv(results / "fleet_types.csv", index=False)
    hourly.loc[hourly.y.isna(), ["time_local", "quality"]].to_csv(results / "unknown_hours.csv")
    prepare_calendar(raw / "calendario.csv").to_csv(processed / "calendar_2022.csv", index=False)
    return hourly


def prepare_calendar(path):
    calendar = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    dates = pd.to_datetime(calendar.iloc[:, 0], format="%d/%m/%Y", errors="raise")
    result = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"), "year": dates.dt.year})
    result["holiday"] = calendar.iloc[:, 2].str.lower().str.contains("festivo", na=False).astype(int)
    result["holiday_name"] = calendar.iloc[:, 4]
    result = result.loc[result.year.eq(2022)].drop(columns="year")
    expected = pd.date_range("2022-01-01", "2022-12-31").strftime("%Y-%m-%d")
    if result.date.duplicated().any() or set(result.date) != set(expected):
        raise ValueError("El calendario no cubre los 365 días de 2022 una sola vez.")
    return result


def audit_weather(root):
    """Auditoría de cobertura de las fuentes opcionales; no entra en los modelos.

    No se interpreta H01..H24 como UTC o local sin resolver antes su semántica.
    La tasa válida se calcula sobre estaciones × días × 24 de cada mes.
    """
    parts = [pd.read_csv(p, sep=";") for p in sorted((root / "data/raw/meteo").glob("*.csv"))]
    weather = pd.concat(parts, ignore_index=True)
    rows = []
    for (month, magnitude), group in weather.groupby(["MES", "MAGNITUD"]):
        flags = group[[f"V{h:02}" for h in range(1, 25)]]
        values = group[[f"H{h:02}" for h in range(1, 25)]].apply(pd.to_numeric, errors="coerce")
        valid = flags.eq("V").to_numpy() & values.notna().to_numpy()
        stations = int(group.ESTACION.nunique())
        days = pd.Period(f"2022-{int(month):02}").days_in_month
        denominator = stations * days * 24
        rows.append({"month": int(month), "magnitude": int(magnitude), "stations": stations,
                     "station_day_rows": len(group), "valid_hours": int(valid.sum()),
                     "expected_hours_active_stations": denominator,
                     "coverage": float(valid.sum() / denominator)})
    pd.DataFrame(rows).to_csv(root / "results/weather_coverage.csv", index=False)
