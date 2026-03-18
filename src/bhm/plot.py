import json
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from dvclive.live import Live


COLORS = {
    "Hessian": (204 / 255, 57 / 255, 42 / 255),
    "GGN": (79 / 255, 155 / 255, 143 / 255),
    "B-GGN": (217 / 255, 116 / 255, 89 / 255),
    "EShampoo": (51 / 255, 102 / 255, 153 / 255),
    "EK-FAC": (228 / 255, 197 / 255, 119 / 255),
    "ETKFAC": (204 / 255, 153 / 255, 102 / 255),
    "Shampoo": (102 / 255, 153 / 255, 204 / 255),
    "K-FAC": (155 / 255, 106 / 255, 145 / 255),
    "TKFAC": (153 / 255, 102 / 255, 51 / 255),
}

METHODS = ["Hessian", "GGN", "B-GGN", "EShampoo", "EK-FAC", "ETKFAC", "Shampoo", "K-FAC", "TKFAC"]

METHOD_KEYS = {
    "Hessian": "hessian_error",
    "GGN": "ggn_error",
    "B-GGN": "block_ggn_error",
    "EShampoo": "eshampoo_error",
    "EK-FAC": "ekfac_error",
    "ETKFAC": "etkfac_error",
    "Shampoo": "shampoo_error",
    "K-FAC": "kfac_error",
    "TKFAC": "tkfac_error",
}

SWEEPS: dict[str, dict[str, object]] = {
    "epoch": {
        "x_key": "experiment.epochs",
        "x_label": "Training Epochs",
        "title_var": "Epoch",
        "fixed": {"experiment.depth": 8, "experiment.width": 16},
        "x_values": [10, 100, 1000],
    },
    "depth": {
        "x_key": "experiment.depth",
        "x_label": "Network Depth",
        "title_var": "Depth",
        "fixed": {"experiment.width": 16, "experiment.epochs": 100},
        "x_values": [1, 4, 8],
    },
    "width": {
        "x_key": "experiment.width",
        "x_label": "Network Width",
        "title_var": "Width",
        "fixed": {"experiment.depth": 1, "experiment.epochs": 100},
        "x_values": [32, 64, 128],
    },
}


def collect_experiments() -> list[dict[str, object]]:
    result = subprocess.run(
        ["uv", "run", "dvc", "exp", "show", "--json", "--no-pager"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"dvc exp show failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(result.stdout)

    experiments: list[dict[str, object]] = []
    for _branch, branch_data in raw.items():
        for exp_id, exp_data in branch_data.items():
            data = exp_data.get("data", {})
            params = data.get("params", {})
            metrics = data.get("metrics", {})

            params_yaml = params.get("params.yaml", {}).get("data", {})
            metrics_json = metrics.get("dvclive/evaluate/metrics.json", {}).get("data", {})

            if not params_yaml or not metrics_json:
                continue

            flat: dict[str, object] = {}
            flat["experiment.depth"] = params_yaml.get("experiment", {}).get("depth")
            flat["experiment.width"] = params_yaml.get("experiment", {}).get("width")
            flat["experiment.epochs"] = params_yaml.get("experiment", {}).get("epochs")
            for key in METHOD_KEYS.values():
                flat[key] = metrics_json.get(key)
            flat["train_loss"] = metrics_json.get("train_loss")

            if flat["hessian_error"] is not None:
                experiments.append(flat)

    return experiments


def filter_sweep(
    experiments: list[dict[str, object]],
    sweep_name: str,
) -> list[dict[str, object]]:
    sweep = SWEEPS[sweep_name]
    fixed = sweep["fixed"]
    x_values = sweep["x_values"]
    x_key = sweep["x_key"]

    matching = []
    for exp in experiments:
        if all(exp.get(k) == v for k, v in fixed.items()):  # type: ignore[union-attr]
            matching.append(exp)

    result = []
    for xv in x_values:  # type: ignore[union-attr]
        for exp in matching:
            if exp.get(x_key) == xv:  # type: ignore[arg-type]
                result.append(exp)
                break

    return result


def make_figure(
    results: list[dict[str, object]],
    sweep_name: str,
) -> plt.Figure:  # type: ignore[name-defined]
    sweep = SWEEPS[sweep_name]
    x_key = str(sweep["x_key"])
    x_label = str(sweep["x_label"])
    title_var = str(sweep["title_var"])

    x_values = [r[x_key] for r in results]
    x_labels = [f"{title_var} {v}" for v in x_values]

    fig, ax = plt.subplots(figsize=(9.0, 3.5))

    n_groups = len(x_values)
    bar_width = 0.08
    n_methods = len(METHODS)
    group_width = n_methods * bar_width
    x_pos = np.arange(n_groups)

    for j, method in enumerate(METHODS):
        key = METHOD_KEYS[method]
        values = [float(r[key]) for r in results]  # type: ignore[arg-type]
        offsets = x_pos - group_width / 2 + bar_width * (j + 0.5)
        ax.bar(
            offsets,
            values,
            bar_width,
            label=method,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Approximation Error")
    ax.set_xlabel(x_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title(f"Approximation Error vs. {x_label} (Digits)")

    fig.tight_layout()
    return fig


def main() -> None:
    experiments = collect_experiments()
    if not experiments:
        print("No experiments found. Run experiments first with:", file=sys.stderr)
        print("  dvc exp run -S experiment=<config>", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(experiments)} experiment(s)")

    with Live(dir="dvclive_plots", save_dvc_exp=False, dvcyaml=False) as live:
        for sweep_name in SWEEPS:
            results = filter_sweep(experiments, sweep_name)
            if not results:
                print(f"  Skipping {sweep_name} sweep: no matching experiments")
                continue
            fig = make_figure(results, sweep_name)
            live.log_image(f"approx_error_{sweep_name}.png", fig)
            plt.close(fig)
            print(f"  Saved approx_error_{sweep_name}.png")


if __name__ == "__main__":
    main()
