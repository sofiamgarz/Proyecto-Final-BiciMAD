"""Experimento inicial: referencias, Ridge y una arquitectura MLP repetida."""

import json
import platform
import time
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import sha256
from .features import NUMERIC


def fallback_values(train, frame):
    """Mediana por día de semana y hora; todo calculado solo en entrenamiento."""
    medians = train.groupby(["weekday", "hour"]).y.median()
    keys = pd.MultiIndex.from_frame(frame[["weekday", "hour"]])
    return pd.Series(medians.reindex(keys).to_numpy(), index=frame.index).fillna(train.y.median())


def evaluate(frame, root):
    config = json.loads((root / "config/experiment.json").read_text())
    train = frame.loc[frame.split.eq("train") & frame.eligible]
    valid = frame.loc[frame.split.eq("validation") & frame.eligible]
    if train.empty or valid.empty or train.index.max() >= valid.index.min():
        raise ValueError("Partición temporal inválida.")
    predictions = valid[["time_local", "y", "hour", "weekday", "day_type"]].copy()
    metrics, diagnostics, losses = [], [], []

    def record(name, p_train, p_valid, fallback=0):
        p_train, p_valid = np.maximum(0, p_train), np.maximum(0, p_valid)
        predictions[name] = p_valid
        metrics.append({"model": name, "n_train": len(train), "n_validation": len(valid),
                        "mae_train": mean_absolute_error(train.y, p_train),
                        "rmse_train": root_mean_squared_error(train.y, p_train),
                        "mae_validation": mean_absolute_error(valid.y, p_valid),
                        "rmse_validation": root_mean_squared_error(valid.y, p_valid),
                        "fallback_validation": fallback})

    for lag in [1, 168]:
        p_train = train[f"lag_{lag}"].fillna(fallback_values(train, train))
        p_valid = valid[f"lag_{lag}"].fillna(fallback_values(train, valid))
        record(f"naive_{lag}", p_train, p_valid, int(valid[f"lag_{lag}"].isna().sum()))
    ridge = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=config["ridge_alpha"]))
    with threadpool_limits(limits=1):
        ridge.fit(train[NUMERIC], train.y)
    record("ridge", ridge.predict(train[NUMERIC]), ridge.predict(valid[NUMERIC]))
    pd.DataFrame({"feature": ridge[:-1].get_feature_names_out(NUMERIC), "coefficient_scaled_x": ridge[-1].coef_}).to_csv(root / "results/ridge_coefficients.csv", index=False)

    for seed in config["mlp_seeds"]:
        # Sin parada temprana aleatoria: se usa la pérdida de entrenamiento.
        net = MLPRegressor(hidden_layer_sizes=tuple(config["mlp_hidden_layers"]),
                           activation="relu", solver="adam", alpha=config["mlp_alpha"],
                           batch_size=config["batch_size"], learning_rate_init=config["mlp_learning_rate"],
                           max_iter=config["mlp_max_iter"], random_state=seed, shuffle=False,
                           early_stopping=False, tol=config["tol"], n_iter_no_change=config["n_iter_no_change"])
        pipeline = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), net)
        model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train[NUMERIC], train.y)
        fitted = model.regressor_[-1]
        record(f"mlp_{seed}", model.predict(train[NUMERIC]), model.predict(valid[NUMERIC]))
        diagnostics.append({"seed": seed, "iterations": int(fitted.n_iter_), "loss_scaled": float(fitted.loss_),
                            "seconds": time.perf_counter() - started,
                            "warnings": [str(w.message) for w in caught]})
        losses += [{"seed": seed, "epoch": i + 1, "loss_scaled": value} for i, value in enumerate(fitted.loss_curve_)]
        print(f"MLP semilla {seed}: {fitted.n_iter_} iteraciones", flush=True)

    results = root / "results"
    table = pd.DataFrame(metrics)
    table.to_csv(results / "metrics_validation.csv", index=False)
    predictions.to_csv(results / "predictions_validation.csv")
    pd.DataFrame(losses).to_csv(results / "mlp_loss.csv", index=False)
    (results / "training_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    errors = []
    for model in table.model:
        for column in ["hour", "day_type"]:
            for label, subset in predictions.groupby(column):
                errors.append({"model": model, "group_by": column, "group": label, "n": len(subset),
                               "mae": mean_absolute_error(subset.y, subset[model]),
                               "rmse": root_mean_squared_error(subset.y, subset[model])})
    pd.DataFrame(errors).to_csv(results / "errors_by_group.csv", index=False)
    run = {"python": platform.python_version(), "platform": platform.platform(),
           "packages": {x: version(x) for x in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "threadpoolctl"]},
           "features": NUMERIC, "configuration": config,
           "processed_sha256": sha256(root / "data/processed/hourly_2022.csv"),
           "experiment_sha256": sha256(root / "config/experiment.json"),
           "train_first_utc": str(train.index.min()), "train_last_utc": str(train.index.max()),
           "validation_first_utc": str(valid.index.min()), "validation_last_utc": str(valid.index.max()),
           "test_evaluated": False, "protocol": "rolling one-hour-ahead; observed past; fixed fitted models"}
    (results / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print(table.to_string(index=False), flush=True)
    return table
