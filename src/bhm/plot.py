import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "Hessian": (204 / 255, 57 / 255, 42 / 255),
    "GGN": (79 / 255, 155 / 255, 143 / 255),
}

METHODS = ["Hessian", "GGN"]

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
            metrics_json = metrics.get("results/metrics.json", {}).get("data", {})

            if not params_yaml or not metrics_json:
                continue

            flat: dict[str, object] = {}
            flat["experiment.depth"] = params_yaml.get("experiment", {}).get("depth")
            flat["experiment.width"] = params_yaml.get("experiment", {}).get("width")
            flat["experiment.epochs"] = params_yaml.get("experiment", {}).get("epochs")
            flat["hessian_error"] = metrics_json.get("hessian_error")
            flat["ggn_error"] = metrics_json.get("ggn_error")
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
    x_key = sweep["x_key"]
    x_values = sweep["x_values"]

    matching = []
    for exp in experiments:
        if all(exp.get(k) == v for k, v in fixed.items()):  # type: ignore[union-attr]
            matching.append(exp)

    # Sort by x_key and deduplicate (keep first match per x_value)
    result = []
    for xv in x_values:  # type: ignore[union-attr]
        for exp in matching:
            if exp.get(x_key) == xv:  # type: ignore[arg-type]
                result.append(exp)
                break

    return result


def plot_approx_error(
    results: list[dict[str, object]],
    sweep_name: str,
    output_path: str,
) -> None:
    sweep = SWEEPS[sweep_name]
    x_key = str(sweep["x_key"])
    x_label = str(sweep["x_label"])
    title_var = str(sweep["title_var"])

    x_values = [r[x_key] for r in results]
    x_labels = [f"{title_var} {v}" for v in x_values]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    n_groups = len(x_values)
    n_methods = len(METHODS)
    bar_width = 0.25
    group_width = n_methods * bar_width
    x_pos = np.arange(n_groups)

    for j, method in enumerate(METHODS):
        key = "hessian_error" if method == "Hessian" else "ggn_error"
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
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    Path("figures").mkdir(exist_ok=True)

    experiments = collect_experiments()
    if not experiments:
        print("No experiments found. Run experiments first with:", file=sys.stderr)
        print("  dvc exp run -S experiment=<config>", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(experiments)} experiment(s)")

    for sweep_name in SWEEPS:
        results = filter_sweep(experiments, sweep_name)
        if not results:
            print(f"  Skipping {sweep_name} sweep: no matching experiments")
            continue
        output = f"figures/approx_error_{sweep_name}.png"
        plot_approx_error(results, sweep_name, output)


if __name__ == "__main__":
    main()
