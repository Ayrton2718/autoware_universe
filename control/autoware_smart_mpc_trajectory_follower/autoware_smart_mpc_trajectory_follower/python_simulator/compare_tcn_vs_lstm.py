# Copyright 2024 Proxima Technology Inc, TIER IV
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare TCN vs LSTM models in the smart-MPC Python simulator.

Run from the python_simulator/ directory::

    python compare_tcn_vs_lstm.py [--t_collect SECONDS] [--skip_data_collection]

Both models are trained on the same pure-pursuit figure-eight dataset, then
evaluated with the MPC controller on the same test scenario.  Results are
saved to ``comparison_results/``.

Outputs
-------
comparison_results/
  train_data/              pure-pursuit training trajectories
  val_data/                pure-pursuit validation trajectories
  lstm/                    LSTM model files + training plots
  lstm_eval/               LSTM MPC evaluation traces
  tcn/                     TCN model files + training plots
  tcn_eval/                TCN MPC evaluation traces
  comparison.png           side-by-side performance bar chart
  metrics.json             numeric results and training times
"""

import argparse
import json
import os
import time
from importlib import reload as ir
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compare TCN vs LSTM in smart-MPC simulator")
parser.add_argument(
    "--t_collect",
    type=float,
    default=300.0,
    help="Duration (s) for data collection sims.  900 gives better training data.",
)
parser.add_argument(
    "--skip_data_collection",
    action="store_true",
    help="Reuse existing train_data / val_data directories (skip data collection).",
)
parser.add_argument(
    "--skip_models",
    nargs="*",
    metavar="MODEL",
    default=[],
    help="Skip training+eval for listed models (lstm, tcn).  E.g. --skip_models lstm",
)
args = parser.parse_args()

ROOT = "comparison_results"

# Locate the trained_model_param.yaml that drive_functions (re-)reads on each ir().
# drive_controller.__init__ calls ir(drive_functions) which re-reads this YAML, so we
# must write the correct use_tcn_for_training value into it before each evaluation run.
_YAML_PATH = Path(__file__).parent / ".." / "param" / "trained_model_param.yaml"
_YAML_PATH = _YAML_PATH.resolve()


def _set_yaml_tcn_flag(use_tcn: bool) -> None:
    """Patch use_tcn_for_training in trained_model_param.yaml in-place."""
    with open(_YAML_PATH) as f:
        doc = yaml.safe_load(f)
    doc["trained_model_parameter"]["memory_for_training"]["use_tcn_for_training"] = use_tcn
    with open(_YAML_PATH, "w") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
TRAIN_DIR = os.path.join(ROOT, "train_data")
VAL_DIR = os.path.join(ROOT, "val_data")
os.makedirs(ROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# Create sim_setting.json before importing python_simulator so that
# perturbed_sim_flag is True and drive_sim() returns performance metrics.
# ---------------------------------------------------------------------------
with open("sim_setting.json", "w") as _f:
    json.dump({}, _f)

# ---------------------------------------------------------------------------
# Import simulator modules (after sim_setting.json has been written).
# ---------------------------------------------------------------------------
from assets import ControlType  # noqa: E402  # type: ignore
import python_simulator  # noqa: E402  # type: ignore
from autoware_smart_mpc_trajectory_follower.scripts import drive_functions  # noqa: E402
import autoware_smart_mpc_trajectory_follower.training_and_data_check.train_drive_NN_model as tdnm  # noqa: E402

INITIAL_ERROR = np.array(
    [0.001, 0.03, 0.01, 0.0, 0, python_simulator.measurement_steer_bias]
)

# ---------------------------------------------------------------------------
# Step 1: Collect shared training / validation data
# ---------------------------------------------------------------------------
if args.skip_data_collection and os.path.isdir(TRAIN_DIR) and os.path.isdir(VAL_DIR):
    print("\n[INFO] Skipping data collection — reusing existing directories.")
else:
    print("\n" + "=" * 60)
    print(f"  COLLECTING DATA  (t_range=[0, {args.t_collect:.0f}] s)")
    print("=" * 60)

    print("\n-- Training trajectories --")
    python_simulator.drive_sim(
        seed=0,
        t_range=[0, args.t_collect],
        control_type=ControlType.pp_eight,
        save_dir=TRAIN_DIR,
        initial_error=INITIAL_ERROR,
    )

    print("\n-- Validation trajectories --")
    python_simulator.drive_sim(
        seed=1,
        t_range=[0, args.t_collect],
        control_type=ControlType.pp_eight,
        save_dir=VAL_DIR,
        initial_error=INITIAL_ERROR,
    )

# ---------------------------------------------------------------------------
# Step 2: Train and evaluate each model type
# ---------------------------------------------------------------------------
METRIC_NAMES = [
    "max_lateral_dev [m]",
    "max_vel_error [m/s]",
    "max_yaw_error [rad]",
    "max_acc_error [m/s2]",
    "max_steer_error [rad]",
    "mean_lateral_dev [m]",
    "mean_vel_error [m/s]",
    "mean_yaw_error [rad]",
    "mean_acc_error [m/s2]",
    "mean_steer_error [rad]",
    "straight_max_lateral_dev [m]",
]

results = {}
train_times = {}

MODEL_CONFIGS = [
    # (label, use_tcn, model-specific kwargs for get_trained_model)
    (
        "lstm",
        False,
        {"hidden_layer_sizes": (16, 16), "hidden_layer_lstm": 8},
    ),
    (
        "tcn",
        True,
        {
            "hidden_layer_sizes": (16, 16),
            "hidden_size_tcn": 64,
            "kernel_size_tcn": 3,
            "n_blocks_tcn": 4,
        },
    ),
]

for model_label, use_tcn, model_kwargs in MODEL_CONFIGS:
    print("\n" + "=" * 60)
    print(f"  {model_label.upper()} MODEL")
    print("=" * 60)

    MODEL_DIR = os.path.join(ROOT, model_label)
    EVAL_DIR = os.path.join(ROOT, model_label + "_eval")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Write the TCN flag into the YAML so that any ir(drive_functions) call
    # (including the one inside drive_controller.__init__) reads the right value.
    _set_yaml_tcn_flag(use_tcn)

    # Reload drive_functions from the updated YAML, then patch in-memory too.
    ir(drive_functions)
    drive_functions.use_memory_for_training = True
    drive_functions.use_tcn_for_training = use_tcn

    # reload the facade module so it re-selects the right trainer class
    ir(tdnm)

    # -- skip training if requested and model already exists --
    if model_label in args.skip_models and os.path.isfile(
        os.path.join(MODEL_DIR, "model_for_test_drive.pth")
    ):
        print(f"\n  [SKIP] Reusing existing model in {MODEL_DIR}/")
        train_times[model_label] = float("nan")
    else:
        # -- train --
        print(f"\n  Training {model_label.upper()}...")
        t0 = time.time()
        trainer = tdnm.train_drive_NN_model(
            alpha_1_for_polynomial_regression=1e-5,
            alpha_2_for_polynomial_regression=1e-5,
        )
        trainer.add_data_from_csv(TRAIN_DIR, add_mode="as_train")
        trainer.add_data_from_csv(VAL_DIR, add_mode="as_val")
        trainer.get_trained_model(
            use_polynomial_reg=True,
            use_selected_polynomial=True,
            batch_size=64,   # larger batches for faster epochs
            patience=5,      # fewer patience epochs to keep comparison run short
            **model_kwargs,
        )
        train_times[model_label] = time.time() - t0
        print(f"  Training time: {train_times[model_label]:.1f} s")

        trainer.plot_trained_result(save_dir=MODEL_DIR)
        trainer.save_models(save_dir=MODEL_DIR)
        print(f"  Model saved to: {MODEL_DIR}/")

    # -- evaluate --
    print(f"\n  Evaluating {model_label.upper()} with MPC controller...")
    metrics = python_simulator.drive_sim(
        save_dir=EVAL_DIR,
        load_dir=MODEL_DIR,
        use_trained_model=True,
        use_trained_model_diff=True,
        initial_error=INITIAL_ERROR,
    )

    if metrics is None:
        print("  WARNING: drive_sim() returned None — metrics not available.")
        results[model_label] = [float("nan")] * len(METRIC_NAMES)
    else:
        results[model_label] = list(metrics)
        print(f"  max lateral deviation : {metrics[0] * 100:.2f} cm")
        print(f"  mean lateral deviation: {metrics[5] * 100:.2f} cm")

# ---------------------------------------------------------------------------
# Step 3: Print comparison table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  COMPARISON RESULTS: LSTM  vs  TCN")
print("=" * 70)
print(f"\n  Training time:  LSTM = {train_times.get('lstm', float('nan')):.1f} s"
      f"   TCN = {train_times.get('tcn', float('nan')):.1f} s")
print(f"  Speed-up:  {'%.2fx' % (train_times.get('lstm', 1) / max(train_times.get('tcn', 1), 1e-9))}"
      " (positive = TCN faster)\n")

header = f"  {'Metric':<32} {'LSTM':>12} {'TCN':>12} {'Better':>8}"
print(header)
print("  " + "-" * 68)
for i, name in enumerate(METRIC_NAMES):
    lstm_val = results.get("lstm", [float("nan")] * len(METRIC_NAMES))[i]
    tcn_val = results.get("tcn", [float("nan")] * len(METRIC_NAMES))[i]
    if not (lstm_val != lstm_val) and not (tcn_val != tcn_val):  # neither is NaN
        better = "LSTM" if lstm_val < tcn_val else ("TCN" if tcn_val < lstm_val else "tie")
    else:
        better = "N/A"
    print(f"  {name:<32} {lstm_val:>12.5f} {tcn_val:>12.5f} {better:>8}")

# ---------------------------------------------------------------------------
# Step 4: Save metrics to JSON
# ---------------------------------------------------------------------------
summary = {
    "t_collect_seconds": args.t_collect,
    "training_times_seconds": train_times,
    "metrics": {
        label: dict(zip(METRIC_NAMES, vals)) for label, vals in results.items()
    },
}
metrics_path = os.path.join(ROOT, "metrics.json")
with open(metrics_path, "w") as _f:
    json.dump(summary, _f, indent=2)
print(f"\n  Saved metrics  → {metrics_path}")

# ---------------------------------------------------------------------------
# Step 5: Generate comparison bar chart
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt  # type: ignore

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)
    fig.suptitle(
        "TCN vs LSTM: MPC Tracking Errors\n"
        f"(training data: {args.t_collect:.0f} s, pure-pursuit figure-eight)",
        fontsize=12,
    )

    bar_colors = {"lstm": "steelblue", "tcn": "darkorange"}
    labels_short = [n.split(" ")[0] for n in METRIC_NAMES]  # strip units
    x = np.arange(5)
    w = 0.35

    # Left: maximum errors (indices 0-4)
    ax = axes[0]
    for offset, label in zip([-w / 2, w / 2], ["lstm", "tcn"]):
        vals = [results.get(label, [float("nan")] * 11)[i] for i in range(5)]
        bars = ax.bar(x + offset, vals, w, label=label.upper(), color=bar_colors[label])
        for bar, v in zip(bars, vals):
            if v == v:  # not NaN
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=45,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([labels_short[i] for i in range(5)], fontsize=8)
    ax.set_title("Maximum Errors")
    ax.set_ylabel("Error (SI units)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: mean errors (indices 5-9)
    ax = axes[1]
    for offset, label in zip([-w / 2, w / 2], ["lstm", "tcn"]):
        vals = [results.get(label, [float("nan")] * 11)[i] for i in range(5, 10)]
        bars = ax.bar(x + offset, vals, w, label=label.upper(), color=bar_colors[label])
        for bar, v in zip(bars, vals):
            if v == v:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=45,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([labels_short[i] for i in range(5, 10)], fontsize=8)
    ax.set_title("Mean Errors")
    ax.set_ylabel("Error (SI units)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    chart_path = os.path.join(ROOT, "comparison.png")
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"  Saved comparison chart → {chart_path}")

except Exception as e:
    print(f"  WARNING: Could not generate plot ({e})")

print("\n  Done.")
