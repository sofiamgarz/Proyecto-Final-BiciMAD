"""Descriptivos de entrenamiento y gráficos del avance; no explora la prueba."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import TZ


def describe(frame, root):
    train = frame.loc[frame.split.eq("train")]
    observed = train.loc[train.y.notna()]
    result = {"grid_hours": len(frame), "unknown_hours_year": int(frame.y.isna().sum()),
              "train_observed": len(observed), "train_describe": observed.y.describe(percentiles=[.25, .5, .75, .95, .99]).to_dict(),
              "train_correlations": {f"lag_{k}": float(train.y.corr(train[f"lag_{k}"])) for k in [1, 24, 168]},
              "train_hour_peak": int(observed.groupby("hour").y.mean().idxmax()),
              "train_hour_lowest": int(observed.groupby("hour").y.mean().idxmin())}
    (root / "results/eda_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    frame.groupby("split").agg(hours=("y", "size"), observed=("y", "count"), usable=("eligible", "sum")).to_csv(root / "results/splits.csv")
    frame[frame.split.ne("test")].groupby("month").agg(departures=("y", "sum"), observed_hours=("y", "count"), mean=("y", "mean")).to_csv(root / "results/monthly_train_validation.csv")
    observed.groupby(["day_type", "hour"]).y.agg(["mean", "median", "count"]).to_csv(root / "results/profile_train.csv")
    daily = observed.copy()
    daily["date"] = daily.index.tz_convert(TZ).strftime("%Y-%m-%d")
    daily.groupby("date").y.agg(["sum", "count"]).to_csv(root / "results/daily_train.csv")
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#263746", "text.color": "#263746", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 2.55), layout="constrained")
    colors = {"laborable": "#1d5271", "fin_semana": "#b27b40", "festivo": "#6b7e66"}
    for typ in ["laborable", "fin_semana", "festivo"]:
        group = observed.loc[observed.day_type.eq(typ)].groupby("hour").y.mean()
        axes[0].plot(group.index, group.values, label=typ.replace("_", " "), color=colors[typ], linewidth=1.6)
    axes[0].set(xlabel="Hora local", ylabel="Salidas medias por hora", xticks=[0, 6, 12, 18, 23])
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(observed.y, bins=32, color="#1d5271", alpha=.85)
    axes[1].set(xlabel="Salidas por hora", ylabel="Número de horas")
    for ax in axes: ax.grid(axis="y", alpha=.16)
    fig.savefig(root / "results/figures/eda_train.pdf")
    fig.savefig(root / "results/figures/eda_train.png", dpi=170)
    plt.close(fig)


def plot_validation(root):
    pred = pd.read_csv(root / "results/predictions_validation.csv")
    time = pd.to_datetime(pred.time_utc, utc=True).dt.tz_convert(TZ)
    keep = (time >= pd.Timestamp("2022-09-05", tz=TZ)) & (time < pd.Timestamp("2022-09-12", tz=TZ))
    fig, ax = plt.subplots(figsize=(9.1, 2.2), layout="constrained")
    ax.plot(time[keep], pred.loc[keep, "y"], color="#263746", label="Observado", linewidth=1.4)
    ax.plot(time[keep], pred.loc[keep, "ridge"], color="#b27b40", label="Ridge", linewidth=1, alpha=.9)
    ax.plot(time[keep], pred.loc[keep, "mlp_42"], color="#2f718a", label="MLP (semilla 42)", linewidth=1)
    ax.set(ylabel="Salidas por hora", xlabel="Semana del 5 al 11 de septiembre de 2022")
    ax.legend(frameon=False, ncols=3, fontsize=8)
    ax.grid(axis="y", alpha=.16)
    fig.savefig(root / "results/figures/validation_week.pdf")
    fig.savefig(root / "results/figures/validation_week.png", dpi=170)
    plt.close(fig)
