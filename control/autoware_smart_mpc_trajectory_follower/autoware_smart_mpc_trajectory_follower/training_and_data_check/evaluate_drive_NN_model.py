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

"""Script for evaluating a trained drive model against CSV driving data.

This script loads a previously trained model together with its polynomial
regression information and evaluates the prediction quality against driving
data stored as CSV files.

Run from the ``training_and_data_check`` directory::

    python3 evaluate_drive_NN_model.py <data_dir> [--model_dir <model_dir>] [--save_dir <save_dir>]

Alternatively the same steps can be reproduced in any Python environment::

    from autoware_smart_mpc_trajectory_follower.training_and_data_check import train_drive_NN_model
    model_evaluator = train_drive_NN_model.train_drive_NN_model()
    model_evaluator.add_data_from_csv(data_dir)
    model_evaluator.load_models(save_dir=model_dir)
    model_evaluator.plot_trained_result(save_dir=save_dir)

Arguments:
    data_dir        Path to the directory containing the CSV driving data.
    --model_dir     Path to the directory containing the model files
                    (``model_for_test_drive.pth`` and
                    ``polynomial_reg_info.npz``).  Defaults to the home
                    directory (``~``).
    --save_dir      Directory where the evaluation plots are saved.
                    Defaults to the current directory.
"""

import argparse
import os

from autoware_smart_mpc_trajectory_follower.training_and_data_check import train_drive_NN_model


def main() -> None:
    """Load a saved model, add CSV data, and plot the prediction results."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained drive NN model by comparing its predictions "
            "against CSV driving data."
        )
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing CSV driving data to evaluate against.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.expanduser("~"),
        help=(
            "Directory containing model_for_test_drive.pth and "
            "polynomial_reg_info.npz.  Defaults to the home directory."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="Directory where evaluation plots are saved.  Defaults to the current directory.",
    )
    args = parser.parse_args()

    model_evaluator = train_drive_NN_model.train_drive_NN_model()
    model_evaluator.add_data_from_csv(args.data_dir)
    model_evaluator.load_models(save_dir=args.model_dir)
    model_evaluator.plot_trained_result(save_dir=args.save_dir)
    print("Evaluation complete. Results saved to", args.save_dir)


if __name__ == "__main__":
    main()
