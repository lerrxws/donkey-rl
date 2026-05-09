import csv
import os

import matplotlib
import numpy as np

from src.config import GRAPH_DIR_NAME

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values = []

    for row in rows:
        raw_value = row.get(name, "")

        if raw_value == "":
            values.append(np.nan)
            continue

        values.append(float(raw_value))

    return np.asarray(values, dtype=np.float32)


def has_columns(rows: list[dict[str, str]], columns: list[str]) -> bool:
    if not rows:
        return False

    return all(column in rows[0] for column in columns)


def ensure_graph_dir(run_dir: str) -> str:
    graph_dir = os.path.join(run_dir, GRAPH_DIR_NAME)
    os.makedirs(graph_dir, exist_ok=True)
    return graph_dir


def save_line_plot(
    x: np.ndarray,
    series: dict[str, np.ndarray],
    title: str,
    ylabel: str,
    path: str,
) -> str:
    plt.figure(figsize=(10, 5))

    for label, y in series.items():
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel("episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)

    if len(series) > 1:
        plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

    return path


def save_line_with_band(
    x: np.ndarray,
    mean: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    title: str,
    ylabel: str,
    path: str,
    label: str = "mean",
) -> str:
    plt.figure(figsize=(10, 5))
    plt.plot(x, mean, label=label)
    plt.fill_between(x, low, high, alpha=0.2, label="range")
    plt.title(title)
    plt.xlabel("episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

    return path
