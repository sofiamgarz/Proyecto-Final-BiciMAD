"""Ejecutar desde la raíz: python -m bicimad all | download | prepare | train."""

import argparse
from pathlib import Path

import pandas as pd

from .data import audit_weather, download, preprocess
from .eda import describe, plot_validation
from .features import build_features, read_hourly
from .models import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "prepare", "train", "all"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command in ["download", "all"]:
        download(root)
    if args.command in ["prepare", "all"]:
        preprocess(root)
        audit_weather(root)
    if args.command in ["train", "all"]:
        hourly = read_hourly(root / "data/processed/hourly_2022.csv")
        calendar = pd.read_csv(root / "data/processed/calendar_2022.csv")
        frame = build_features(hourly, calendar)
        frame.to_csv(root / "data/processed/features_2022.csv")
        describe(frame, root)
        evaluate(frame, root)
        plot_validation(root)


if __name__ == "__main__":
    main()
