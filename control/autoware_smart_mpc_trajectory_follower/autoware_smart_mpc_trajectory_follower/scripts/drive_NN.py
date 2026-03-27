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


from autoware_smart_mpc_trajectory_follower import proxima_calc
from autoware_smart_mpc_trajectory_follower.scripts import drive_functions
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

dim_steer_layer_1_head = 32
dim_steer_layer_1_tail = 16
dim_steer_layer_2 = 8
dim_acc_layer_1 = 16
dim_acc_layer_2 = 16

loss_weights = torch.tensor(
    [
        drive_functions.NN_x_weight,
        drive_functions.NN_y_weight,
        drive_functions.NN_yaw_weight,
        drive_functions.NN_v_weight,
        drive_functions.NN_acc_weight,
        drive_functions.NN_steer_weight,
    ]
)
loss_weights_diff = torch.tensor(
    [
        drive_functions.NN_x_weight_diff,
        drive_functions.NN_y_weight_diff,
        drive_functions.NN_yaw_weight_diff,
        drive_functions.NN_v_weight_diff,
        drive_functions.NN_acc_weight_diff,
        drive_functions.NN_steer_weight_diff,
    ]
)
loss_weights_two_diff = torch.tensor(
    [
        drive_functions.NN_x_weight_two_diff,
        drive_functions.NN_y_weight_two_diff,
        drive_functions.NN_yaw_weight_two_diff,
        drive_functions.NN_v_weight_two_diff,
        drive_functions.NN_acc_weight_two_diff,
        drive_functions.NN_steer_weight_two_diff,
    ]
)


def nominal_model_input(Var, lam, step):
    """Calculate prediction with nominal model that takes into account time constants for the first order and time delays related to the input."""
    nominal_pred = Var[:, 0]
    nominal_pred = (
        nominal_pred + (Var[:, step] - nominal_pred) * drive_functions.ctrl_time_step / lam
    )
    nominal_pred = (
        nominal_pred + (Var[:, step - 1] - nominal_pred) * drive_functions.ctrl_time_step / lam
    )
    nominal_pred = (
        nominal_pred + (Var[:, step - 2] - nominal_pred) * drive_functions.ctrl_time_step / lam
    )
    return nominal_pred


def loss_fn_plus_tanh(loss_fn, pred, Y, tanh_gain, tanh_weight):
    """Compute the loss function to be used in the training."""
    loss = loss_fn(pred * loss_weights, Y * loss_weights)
    loss += tanh_weight * loss_fn(
        torch.tanh(tanh_gain * (pred[:, -1] - Y[:, -1])), torch.zeros(Y.shape[0])
    )
    return loss


def loss_fn_plus_tanh_with_memory(
    loss_fn, pred, Y, tanh_gain, tanh_weight, first_order_weight, second_order_weight
):
    """Compute the loss function to be used in the training."""
    loss = loss_fn(pred * loss_weights, Y * loss_weights)
    loss += (
        first_order_weight
        * loss_fn(
            (pred[:, 1:] - pred[:, :-1]) * loss_weights_diff,
            (Y[:, 1:] - Y[:, :-1]) * loss_weights_diff,
        )
        / drive_functions.ctrl_time_step
    )
    loss += (
        second_order_weight
        * loss_fn(
            (pred[:, 2:] + pred[:, :-2] - 2 * pred[:, 1:-1]) * loss_weights_two_diff,
            (Y[:, 2:] + Y[:, :-2] - 2 * Y[:, 1:-1]) * loss_weights_two_diff,
        )
        / (drive_functions.ctrl_time_step * drive_functions.ctrl_time_step)
    )
    loss += tanh_weight * loss_fn(
        torch.tanh(tanh_gain * (pred[:, :, -1] - Y[:, :, -1])),
        torch.zeros((Y.shape[0], Y.shape[1])),
    )
    return loss


# example usage: `model = DriveNeuralNetwork(params...).to("cpu")`
class DriveNeuralNetwork(nn.Module):
    """Define the neural net model to be used in vehicle control."""

    def __init__(
        self,
        hidden_layer_sizes=(32, 16),
        randomize=0.01,
        acc_drop_out=0.0,
        steer_drop_out=0.0,
        acc_delay_step=drive_functions.acc_delay_step,
        steer_delay_step=drive_functions.steer_delay_step,
        acc_time_constant_ctrl=drive_functions.acc_time_constant,
        steer_time_constant_ctrl=drive_functions.steer_time_constant,
        acc_queue_size=drive_functions.acc_ctrl_queue_size,
        steer_queue_size=drive_functions.steer_ctrl_queue_size,
        steer_queue_size_core=drive_functions.steer_ctrl_queue_size_core,
    ):
        super().__init__()
        self.acc_time_constant_ctrl = acc_time_constant_ctrl
        self.acc_delay_step = acc_delay_step
        self.steer_time_constant_ctrl = steer_time_constant_ctrl
        self.steer_delay_step = steer_delay_step

        lb = -randomize
        ub = randomize
        self.acc_input_index = np.concatenate(([1], np.arange(acc_queue_size) + 3))
        self.steer_input_index = np.concatenate(
            ([2], np.arange(steer_queue_size_core) + acc_queue_size + 3)
        )
        self.steer_input_index_full = np.arange(steer_queue_size) + acc_queue_size + 3
        self.acc_layer_1 = nn.Sequential(
            nn.Linear(self.acc_input_index.shape[0], dim_acc_layer_1),
            nn.ReLU(),
        )
        nn.init.uniform_(self.acc_layer_1[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.acc_layer_1[0].bias, a=lb, b=ub)
        self.steer_layer_1_head = nn.Sequential(
            nn.Linear(self.steer_input_index.shape[0], dim_steer_layer_1_head),
            nn.ReLU(),
        )
        nn.init.uniform_(self.steer_layer_1_head[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.steer_layer_1_head[0].bias, a=lb, b=ub)

        self.steer_layer_1_tail = nn.Sequential(
            nn.Linear(self.steer_input_index_full.shape[0], dim_steer_layer_1_tail), nn.ReLU()
        )
        nn.init.uniform_(self.steer_layer_1_tail[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.steer_layer_1_tail[0].bias, a=lb, b=ub)

        self.acc_layer_2 = nn.Sequential(nn.Linear(dim_acc_layer_1, dim_acc_layer_2), nn.ReLU())
        nn.init.uniform_(self.acc_layer_2[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.acc_layer_2[0].bias, a=lb, b=ub)

        self.steer_layer_2 = nn.Sequential(
            nn.Linear(dim_steer_layer_1_head + dim_steer_layer_1_tail, dim_steer_layer_2), nn.ReLU()
        )
        nn.init.uniform_(self.steer_layer_2[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.steer_layer_2[0].bias, a=lb, b=ub)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1 + dim_acc_layer_2 + dim_steer_layer_2, hidden_layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.ReLU(),
        )
        nn.init.uniform_(self.linear_relu_stack[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.linear_relu_stack[0].bias, a=lb, b=ub)
        nn.init.uniform_(self.linear_relu_stack[2].weight, a=lb, b=ub)
        nn.init.uniform_(self.linear_relu_stack[2].bias, a=lb, b=ub)

        self.finalize = nn.Linear(hidden_layer_sizes[1] + dim_acc_layer_2 + dim_steer_layer_2, 6)

        nn.init.uniform_(self.finalize.weight, a=lb, b=ub)
        nn.init.uniform_(self.finalize.bias, a=lb, b=ub)

        self.acc_dropout = nn.Dropout(acc_drop_out)
        self.steer_dropout = nn.Dropout(steer_drop_out)

    def forward(self, x):
        acc_layer_1 = self.acc_layer_1(drive_functions.acc_normalize * x[:, self.acc_input_index])
        steer_layer_1 = torch.cat(
            (
                self.steer_layer_1_head(
                    drive_functions.steer_normalize * x[:, self.steer_input_index]
                ),
                self.steer_layer_1_tail(
                    drive_functions.steer_normalize * x[:, self.steer_input_index_full]
                ),
            ),
            dim=1,
        )
        acc_layer_2 = self.acc_layer_2(self.acc_dropout(acc_layer_1))
        steer_layer_2 = self.steer_layer_2(self.steer_dropout(steer_layer_1))

        pre_pred = self.linear_relu_stack(
            torch.cat(
                (drive_functions.vel_normalize * x[:, [0]], acc_layer_2, steer_layer_2), dim=1
            )
        )
        pred = self.finalize(torch.cat((pre_pred, acc_layer_2, steer_layer_2), dim=1))
        return pred


class DriveNeuralNetworkWithMemory(nn.Module):
    """Define the neural net model with memory to be used in vehicle control."""

    def __init__(
        self,
        hidden_layer_sizes=(32, 16),
        hidden_layer_lstm=64,
        randomize=0.01,
        acc_drop_out=0.0,
        steer_drop_out=0.0,
        acc_delay_step=drive_functions.acc_delay_step,
        steer_delay_step=drive_functions.steer_delay_step,
        acc_time_constant_ctrl=drive_functions.acc_time_constant,
        steer_time_constant_ctrl=drive_functions.steer_time_constant,
        acc_queue_size=drive_functions.acc_ctrl_queue_size,
        steer_queue_size=drive_functions.steer_ctrl_queue_size,
        steer_queue_size_core=drive_functions.steer_ctrl_queue_size_core,
    ):
        super().__init__()
        self.acc_time_constant_ctrl = acc_time_constant_ctrl
        self.acc_delay_step = acc_delay_step
        self.steer_time_constant_ctrl = steer_time_constant_ctrl
        self.steer_delay_step = steer_delay_step

        lb = -randomize
        ub = randomize
        self.acc_input_index = np.concatenate(([1], np.arange(acc_queue_size) + 3))
        self.steer_input_index = np.concatenate(
            ([2], np.arange(steer_queue_size_core) + acc_queue_size + 3)
        )
        self.steer_input_index_full = np.arange(steer_queue_size) + acc_queue_size + 3
        self.acc_layer_1 = nn.Sequential(
            nn.Linear(self.acc_input_index.shape[0], dim_acc_layer_1),
            nn.ReLU(),
        )
        nn.init.uniform_(self.acc_layer_1[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.acc_layer_1[0].bias, a=lb, b=ub)
        self.steer_layer_1_head = nn.Sequential(
            nn.Linear(self.steer_input_index.shape[0], dim_steer_layer_1_head),
            nn.ReLU(),
        )
        nn.init.uniform_(self.steer_layer_1_head[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.steer_layer_1_head[0].bias, a=lb, b=ub)

        self.steer_layer_1_tail = nn.Sequential(
            nn.Linear(self.steer_input_index_full.shape[0], dim_steer_layer_1_tail), nn.ReLU()
        )
        nn.init.uniform_(self.steer_layer_1_tail[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.steer_layer_1_tail[0].bias, a=lb, b=ub)

        self.acc_layer_2 = nn.Sequential(nn.Linear(dim_acc_layer_1, dim_acc_layer_2), nn.ReLU())
        nn.init.uniform_(self.acc_layer_2[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.acc_layer_2[0].bias, a=lb, b=ub)

        self.steer_layer_2 = nn.Sequential(
            nn.Linear(dim_steer_layer_1_head + dim_steer_layer_1_tail, dim_steer_layer_2), nn.ReLU()
        )
        nn.init.uniform_(self.steer_layer_2[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.steer_layer_2[0].bias, a=lb, b=ub)

        self.lstm = nn.LSTM(
            1 + dim_acc_layer_2 + dim_steer_layer_2, hidden_layer_lstm, batch_first=True
        )

        nn.init.uniform_(self.lstm.weight_hh_l0, a=lb, b=ub)
        nn.init.uniform_(self.lstm.weight_ih_l0, a=lb, b=ub)
        nn.init.uniform_(self.lstm.bias_hh_l0, a=lb, b=ub)
        nn.init.uniform_(self.lstm.bias_ih_l0, a=lb, b=ub)

        self.linear_relu_stack_1 = nn.Sequential(
            nn.Linear(1 + dim_acc_layer_2 + dim_steer_layer_2, hidden_layer_sizes[0]),
            nn.ReLU(),
        )
        nn.init.uniform_(self.linear_relu_stack_1[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.linear_relu_stack_1[0].bias, a=lb, b=ub)

        self.linear_relu_stack_2 = nn.Sequential(
            nn.Linear(hidden_layer_lstm + hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.ReLU(),
        )
        nn.init.uniform_(self.linear_relu_stack_2[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.linear_relu_stack_2[0].bias, a=lb, b=ub)

        self.finalize = nn.Linear(hidden_layer_sizes[1] + dim_acc_layer_2 + dim_steer_layer_2, 6)

        nn.init.uniform_(self.finalize.weight, a=lb, b=ub)
        nn.init.uniform_(self.finalize.bias, a=lb, b=ub)

        self.acc_dropout = nn.Dropout(acc_drop_out)
        self.steer_dropout = nn.Dropout(steer_drop_out)

    def forward(self, x):
        acc_layer_1 = self.acc_layer_1(
            drive_functions.acc_normalize * x[:, :, self.acc_input_index]
        )
        steer_layer_1 = torch.cat(
            (
                self.steer_layer_1_head(
                    drive_functions.steer_normalize * x[:, :, self.steer_input_index]
                ),
                self.steer_layer_1_tail(
                    drive_functions.steer_normalize * x[:, :, self.steer_input_index_full]
                ),
            ),
            dim=2,
        )
        acc_layer_2 = self.acc_layer_2(self.acc_dropout(acc_layer_1))
        steer_layer_2 = self.steer_layer_2(self.steer_dropout(steer_layer_1))
        h_1 = torch.cat(
            (drive_functions.vel_normalize * x[:, :, [0]], acc_layer_2, steer_layer_2), dim=2
        )
        pre_pred, _ = self.lstm(h_1)
        pre_pred = torch.cat((pre_pred, self.linear_relu_stack_1(h_1)), dim=2)
        pre_pred = self.linear_relu_stack_2(pre_pred)
        pred = self.finalize(torch.cat((pre_pred, acc_layer_2, steer_layer_2), dim=2))
        return pred


class TCNBlock(nn.Module):
    """Single dilated causal 1D convolution block with residual connection.

    Implements causal convolution by applying left-only padding so the output at
    each timestep depends only on current and past inputs.  A residual (skip)
    connection is added; when input and output channel counts differ a 1x1
    convolution aligns dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
        randomize: float = 0.01,
    ):
        super().__init__()
        # Amount of left-padding needed to keep sequence length unchanged (causal).
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # 1x1 residual projection when channel dims differ.
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        nn.init.uniform_(self.conv.weight, a=-randomize, b=randomize)
        nn.init.uniform_(self.conv.bias, a=-randomize, b=randomize)
        if self.residual_conv is not None:
            nn.init.uniform_(self.residual_conv.weight, a=-randomize, b=randomize)
            nn.init.uniform_(self.residual_conv.bias, a=-randomize, b=randomize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, seq_len)
        padded = F.pad(x, (self.causal_padding, 0))
        out = self.dropout(self.relu(self.conv(padded)))
        residual = self.residual_conv(x) if self.residual_conv is not None else x
        return out + residual


class TCN(nn.Module):
    """Temporal Convolutional Network: stack of dilated causal convolution blocks.

    Each block doubles the dilation, so the receptive field grows exponentially
    with depth.  The receptive field of n_blocks blocks is::

        RF = sum_{i=0}^{n_blocks-1} (kernel_size - 1) * 2^i + 1

    For kernel_size=3, n_blocks=4: RF = 2*(1+2+4+8)+1 = 31 time steps.

    Input/output format mirrors ``nn.LSTM(batch_first=True)``:
    ``(batch, seq_len, features)`` → ``(batch, seq_len, hidden_size)``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int = 3,
        n_blocks: int = 4,
        dropout: float = 0.0,
        randomize: float = 0.01,
    ):
        super().__init__()
        self.receptive_field = sum(
            (kernel_size - 1) * (2**i) for i in range(n_blocks)
        ) + 1
        blocks = []
        for i in range(n_blocks):
            dilation = 2**i
            in_ch = input_size if i == 0 else hidden_size
            blocks.append(
                TCNBlock(in_ch, hidden_size, kernel_size, dilation, dropout, randomize)
            )
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) — same convention as LSTM batch_first=True
        # Conv1d expects (batch, channels, seq_len)
        out = self.network(x.transpose(1, 2))
        return out.transpose(1, 2)  # (batch, seq_len, hidden_size)


class DriveNeuralNetworkWithTCN(nn.Module):
    """Neural net with TCN memory, a drop-in training replacement for DriveNeuralNetworkWithMemory.

    Replaces the single-layer LSTM with a Temporal Convolutional Network built from
    dilated causal convolutions.  All pre-processing layers (acc/steer feature
    extraction) and post-processing layers (linear_relu_stack, finalize) are
    identical to ``DriveNeuralNetworkWithMemory``.

    Input shape:  ``(batch, seq_len, raw_features)``
    Output shape: ``(batch, seq_len, 6)``
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (32, 16),
        hidden_size_tcn: int = 64,
        kernel_size_tcn: int = 3,
        n_blocks_tcn: int = 4,
        randomize: float = 0.01,
        acc_drop_out: float = 0.0,
        steer_drop_out: float = 0.0,
        acc_delay_step=drive_functions.acc_delay_step,
        steer_delay_step=drive_functions.steer_delay_step,
        acc_time_constant_ctrl=drive_functions.acc_time_constant,
        steer_time_constant_ctrl=drive_functions.steer_time_constant,
        acc_queue_size=drive_functions.acc_ctrl_queue_size,
        steer_queue_size=drive_functions.steer_ctrl_queue_size,
        steer_queue_size_core=drive_functions.steer_ctrl_queue_size_core,
    ):
        super().__init__()
        self.acc_time_constant_ctrl = acc_time_constant_ctrl
        self.acc_delay_step = acc_delay_step
        self.steer_time_constant_ctrl = steer_time_constant_ctrl
        self.steer_delay_step = steer_delay_step

        lb = -randomize
        ub = randomize
        self.acc_input_index = np.concatenate(([1], np.arange(acc_queue_size) + 3))
        self.steer_input_index = np.concatenate(
            ([2], np.arange(steer_queue_size_core) + acc_queue_size + 3)
        )
        self.steer_input_index_full = np.arange(steer_queue_size) + acc_queue_size + 3

        # --- Feature extraction layers (identical to WithMemory) ---
        self.acc_layer_1 = nn.Sequential(
            nn.Linear(self.acc_input_index.shape[0], dim_acc_layer_1),
            nn.ReLU(),
        )
        nn.init.uniform_(self.acc_layer_1[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.acc_layer_1[0].bias, a=lb, b=ub)

        self.steer_layer_1_head = nn.Sequential(
            nn.Linear(self.steer_input_index.shape[0], dim_steer_layer_1_head),
            nn.ReLU(),
        )
        nn.init.uniform_(self.steer_layer_1_head[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.steer_layer_1_head[0].bias, a=lb, b=ub)

        self.steer_layer_1_tail = nn.Sequential(
            nn.Linear(self.steer_input_index_full.shape[0], dim_steer_layer_1_tail), nn.ReLU()
        )
        nn.init.uniform_(self.steer_layer_1_tail[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.steer_layer_1_tail[0].bias, a=lb, b=ub)

        self.acc_layer_2 = nn.Sequential(nn.Linear(dim_acc_layer_1, dim_acc_layer_2), nn.ReLU())
        nn.init.uniform_(self.acc_layer_2[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.acc_layer_2[0].bias, a=lb, b=ub)

        self.steer_layer_2 = nn.Sequential(
            nn.Linear(dim_steer_layer_1_head + dim_steer_layer_1_tail, dim_steer_layer_2),
            nn.ReLU(),
        )
        nn.init.uniform_(self.steer_layer_2[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.steer_layer_2[0].bias, a=lb, b=ub)

        # --- TCN replaces LSTM ---
        tcn_input_size = 1 + dim_acc_layer_2 + dim_steer_layer_2  # same as LSTM input size
        self.tcn = TCN(
            input_size=tcn_input_size,
            hidden_size=hidden_size_tcn,
            kernel_size=kernel_size_tcn,
            n_blocks=n_blocks_tcn,
            randomize=randomize,
        )

        # --- Post-TCN fusion layers (mirror WithMemory) ---
        self.linear_relu_stack_1 = nn.Sequential(
            nn.Linear(tcn_input_size, hidden_layer_sizes[0]),
            nn.ReLU(),
        )
        nn.init.uniform_(self.linear_relu_stack_1[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.linear_relu_stack_1[0].bias, a=lb, b=ub)

        self.linear_relu_stack_2 = nn.Sequential(
            nn.Linear(hidden_size_tcn + hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.ReLU(),
        )
        nn.init.uniform_(self.linear_relu_stack_2[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.linear_relu_stack_2[0].bias, a=lb, b=ub)

        self.finalize = nn.Linear(
            hidden_layer_sizes[1] + dim_acc_layer_2 + dim_steer_layer_2, 6
        )
        nn.init.uniform_(self.finalize.weight, a=lb, b=ub)
        nn.init.uniform_(self.finalize.bias, a=lb, b=ub)

        self.acc_dropout = nn.Dropout(acc_drop_out)
        self.steer_dropout = nn.Dropout(steer_drop_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, raw_features)
        acc_layer_1 = self.acc_layer_1(
            drive_functions.acc_normalize * x[:, :, self.acc_input_index]
        )
        steer_layer_1 = torch.cat(
            (
                self.steer_layer_1_head(
                    drive_functions.steer_normalize * x[:, :, self.steer_input_index]
                ),
                self.steer_layer_1_tail(
                    drive_functions.steer_normalize * x[:, :, self.steer_input_index_full]
                ),
            ),
            dim=2,
        )
        acc_layer_2 = self.acc_layer_2(self.acc_dropout(acc_layer_1))
        steer_layer_2 = self.steer_layer_2(self.steer_dropout(steer_layer_1))
        h_1 = torch.cat(
            (drive_functions.vel_normalize * x[:, :, [0]], acc_layer_2, steer_layer_2), dim=2
        )
        # TCN processes the full sequence causally; shape: (batch, seq_len, hidden_size_tcn)
        tcn_out = self.tcn(h_1)
        pre_pred = torch.cat((tcn_out, self.linear_relu_stack_1(h_1)), dim=2)
        pre_pred = self.linear_relu_stack_2(pre_pred)
        pred = self.finalize(torch.cat((pre_pred, acc_layer_2, steer_layer_2), dim=2))
        return pred


class transform_model_with_tcn_to_pred:
    """Python inference wrapper for DriveNeuralNetworkWithTCN.

    Provides the same ``pred`` / ``pred_only_state`` / ``Pred`` interface
    expected by ``drive_controller``, but uses a sliding context buffer instead
    of an LSTM hidden state.  The buffer stores past raw input vectors aligned
    at the MPC time step; at each call the TCN is run on the full buffer and
    only the last-timestep output is returned.

    Note: ``pred_with_diff`` falls back to ``pred_with_poly_diff`` (polynomial
    regression Jacobian) because analytic TCN Jacobians require additional C++
    work.  The MPC optimisation therefore uses the polynomial approximation for
    the gradient, which is the same behaviour as ``reflect_only_poly_diff=True``.
    """

    def __init__(
        self,
        model: "DriveNeuralNetworkWithTCN",
        A_for_linear_reg: np.ndarray,
        b_for_linear_reg: np.ndarray,
        deg: int,
        acc_delay_step: int = drive_functions.acc_delay_step,
        steer_delay_step: int = drive_functions.steer_delay_step,
        acc_queue_size: int = drive_functions.acc_ctrl_queue_size,
        steer_queue_size: int = drive_functions.steer_ctrl_queue_size,
        steer_queue_size_core: int = drive_functions.steer_ctrl_queue_size_core,
        vel_normalize: float = drive_functions.vel_normalize,
        acc_normalize: float = drive_functions.acc_normalize,
        steer_normalize: float = drive_functions.steer_normalize,
    ):
        from sklearn.preprocessing import PolynomialFeatures

        self.model = model
        self.A = A_for_linear_reg
        self.b = b_for_linear_reg
        self.deg = deg
        self.acc_delay_step = acc_delay_step
        self.steer_delay_step = steer_delay_step
        self.acc_queue_size = acc_queue_size
        self.steer_queue_size = steer_queue_size
        self.steer_queue_size_core = steer_queue_size_core
        self.vel_normalize = vel_normalize
        self.acc_normalize = acc_normalize
        self.steer_normalize = steer_normalize
        self.polynomial_features = PolynomialFeatures(degree=deg, include_bias=False)
        # Sliding window of raw input vectors; maximum length = TCN receptive field.
        self._context: list = []
        self._max_context: int = model.tcn.receptive_field

    def reset_context(self) -> None:
        """Clear the context buffer (call when the episode restarts)."""
        self._context = []

    def _poly_correction(self, x_current: np.ndarray) -> np.ndarray:
        """Polynomial regression correction term for a single input vector."""
        from autoware_smart_mpc_trajectory_follower.training_and_data_check import (
            train_drive_NN_model_with_memory,
        )

        ctrl_idx = train_drive_NN_model_with_memory.ctrl_index_for_polynomial_reg
        feats = self.polynomial_features.fit_transform(x_current[ctrl_idx][np.newaxis])
        return (feats @ self.A.T + self.b)[0]

    def _run_tcn(self, x_current: np.ndarray) -> np.ndarray:
        """Add x_current to the context, run TCN, return last-step NN output."""
        self._context.append(x_current.copy())
        if len(self._context) > self._max_context:
            self._context = self._context[-self._max_context :]
        ctx = np.array(self._context, dtype=np.float32)  # (T, features)
        with torch.no_grad():
            x_tensor = torch.from_numpy(ctx[np.newaxis])  # (1, T, features)
            output = self.model(x_tensor)  # (1, T, 6)
        return output[0, -1].numpy()  # last timestep: (6,)

    def pred(self, x_current: np.ndarray) -> np.ndarray:
        """Point prediction: NN output + polynomial correction."""
        return self._run_tcn(x_current) + self._poly_correction(x_current)

    def pred_only_state(self, x_current: np.ndarray) -> np.ndarray:
        """State-only prediction (no derivative); alias for pred."""
        return self.pred(x_current)

    def pred_with_poly_diff(self, x_current: np.ndarray, dx_current: np.ndarray) -> np.ndarray:
        """Prediction with gradient via polynomial Jacobian only (TCN gradient not implemented)."""
        # Reuse last context run without re-appending.
        nn_out = self._run_tcn(x_current)
        from autoware_smart_mpc_trajectory_follower.training_and_data_check import (
            train_drive_NN_model_with_memory,
        )

        ctrl_idx = train_drive_NN_model_with_memory.ctrl_index_for_polynomial_reg
        feats = self.polynomial_features.fit_transform(x_current[ctrl_idx][np.newaxis])
        poly_pred = (feats @ self.A.T + self.b)[0]
        # Jacobian of polynomial w.r.t. ctrl_idx components of x_current.
        poly_feat_deriv = self.polynomial_features.transform(
            (x_current[ctrl_idx] + 1e-5 * dx_current[ctrl_idx])[np.newaxis]
        )
        poly_pred_dx = ((poly_feat_deriv @ self.A.T + self.b)[0] - poly_pred) / 1e-5
        return nn_out + poly_pred, poly_pred_dx

    def Pred(self, X_candidates: np.ndarray) -> np.ndarray:
        """Batch prediction for MPPI candidate trajectories.

        Args:
            X_candidates: shape ``(n_candidates, raw_features)``

        Returns:
            predictions of shape ``(n_candidates, 6)``
        """
        # Build a batch where each candidate reuses the current context.
        ctx = np.array(self._context, dtype=np.float32) if self._context else np.zeros(
            (1, X_candidates.shape[-1]), dtype=np.float32
        )
        # Append each candidate as the "next" step in the same context.
        ctx_expanded = np.tile(ctx[np.newaxis], (X_candidates.shape[0], 1, 1))
        cands = X_candidates[:, np.newaxis, :].astype(np.float32)
        batch_input = np.concatenate([ctx_expanded, cands], axis=1)
        with torch.no_grad():
            x_tensor = torch.from_numpy(batch_input)  # (n, T+1, features)
            output = self.model(x_tensor)  # (n, T+1, 6)
        nn_out = output[:, -1].numpy()  # (n, 6)
        from autoware_smart_mpc_trajectory_follower.training_and_data_check import (
            train_drive_NN_model_with_memory,
        )

        ctrl_idx = train_drive_NN_model_with_memory.ctrl_index_for_polynomial_reg
        feats = self.polynomial_features.fit_transform(X_candidates[:, ctrl_idx])
        poly_out = feats @ self.A.T + self.b
        return nn_out + poly_out


class EarlyStopping:
    """Class for early stopping in NN training."""

    def __init__(self, initial_loss, tol=0.01, patience=30):
        self.epoch = 0  # Initialise the counter for the number of epochs being monitored.
        self.best_loss = float("inf")  # Initialise loss of comparison with infinity 'inf'.
        self.patience = patience  # Initialise the number of epochs to be monitored with a parameter
        self.initial_loss = initial_loss.detach().item()
        self.tol = tol

    def __call__(self, current_loss):
        current_loss_num = current_loss.detach().item()
        if current_loss_num + self.tol * self.initial_loss > self.best_loss:
            self.epoch += 1
        else:
            self.epoch = 0
        if current_loss_num < self.best_loss:
            self.best_loss = current_loss_num
        if self.epoch >= self.patience:
            return True
        return False


class transform_model_to_c:
    """Pass the necessary information to the C++ program to call the trained model at high speed."""

    def __init__(
        self,
        model,
        A_for_linear_reg,
        b_for_linear_reg,
        deg,
        acc_delay_step=drive_functions.acc_delay_step,
        steer_delay_step=drive_functions.steer_delay_step,
        acc_queue_size=drive_functions.acc_ctrl_queue_size,
        steer_queue_size=drive_functions.steer_ctrl_queue_size,
        steer_queue_size_core=drive_functions.steer_ctrl_queue_size_core,
        vel_normalize=drive_functions.vel_normalize,
        acc_normalize=drive_functions.acc_normalize,
        steer_normalize=drive_functions.steer_normalize,
    ):
        self.transform = proxima_calc.transform_model_to_eigen()
        self.transform.set_params(
            model.acc_layer_1[0].weight.detach().numpy().astype(np.float64),
            model.steer_layer_1_head[0].weight.detach().numpy().astype(np.float64),
            model.steer_layer_1_tail[0].weight.detach().numpy().astype(np.float64),
            model.acc_layer_2[0].weight.detach().numpy().astype(np.float64),
            model.steer_layer_2[0].weight.detach().numpy().astype(np.float64),
            model.linear_relu_stack[0].weight.detach().numpy().astype(np.float64),
            model.linear_relu_stack[2].weight.detach().numpy().astype(np.float64),
            model.finalize.weight.detach().numpy().astype(np.float64),
            model.acc_layer_1[0].bias.detach().numpy().astype(np.float64),
            model.steer_layer_1_head[0].bias.detach().numpy().astype(np.float64),
            model.steer_layer_1_tail[0].bias.detach().numpy().astype(np.float64),
            model.acc_layer_2[0].bias.detach().numpy().astype(np.float64),
            model.steer_layer_2[0].bias.detach().numpy().astype(np.float64),
            model.linear_relu_stack[0].bias.detach().numpy().astype(np.float64),
            model.linear_relu_stack[2].bias.detach().numpy().astype(np.float64),
            model.finalize.bias.detach().numpy().astype(np.float64),
            A_for_linear_reg,
            b_for_linear_reg,
            deg,
            acc_delay_step,
            steer_delay_step,
            acc_queue_size,
            steer_queue_size,
            steer_queue_size_core,
            vel_normalize,
            acc_normalize,
            steer_normalize,
        )
        self.pred = self.transform.rot_and_d_rot_error_prediction
        self.pred_only_state = self.transform.rotated_error_prediction
        self.pred_with_diff = self.transform.rot_and_d_rot_error_prediction_with_diff
        self.pred_with_poly_diff = self.transform.rot_and_d_rot_error_prediction_with_poly_diff
        self.Pred = self.transform.Rotated_error_prediction


class transform_model_with_memory_to_c:
    """Pass the necessary information to the C++ program to call the trained model at high speed."""

    def __init__(
        self,
        model,
        A_for_linear_reg,
        b_for_linear_reg,
        deg,
        acc_delay_step=drive_functions.acc_delay_step,
        steer_delay_step=drive_functions.steer_delay_step,
        acc_queue_size=drive_functions.acc_ctrl_queue_size,
        steer_queue_size=drive_functions.steer_ctrl_queue_size,
        steer_queue_size_core=drive_functions.steer_ctrl_queue_size_core,
        vel_normalize=drive_functions.vel_normalize,
        acc_normalize=drive_functions.acc_normalize,
        steer_normalize=drive_functions.steer_normalize,
    ):
        self.transform = proxima_calc.transform_model_with_memory_to_eigen()
        self.transform.set_params(
            model.acc_layer_1[0].weight.detach().numpy().astype(np.float64),
            model.steer_layer_1_head[0].weight.detach().numpy().astype(np.float64),
            model.steer_layer_1_tail[0].weight.detach().numpy().astype(np.float64),
            model.acc_layer_2[0].weight.detach().numpy().astype(np.float64),
            model.steer_layer_2[0].weight.detach().numpy().astype(np.float64),
            model.lstm.weight_ih_l0.detach().numpy().astype(np.float64),
            model.lstm.weight_hh_l0.detach().numpy().astype(np.float64),
            model.linear_relu_stack_1[0].weight.detach().numpy().astype(np.float64),
            model.linear_relu_stack_2[0].weight.detach().numpy().astype(np.float64),
            model.finalize.weight.detach().numpy().astype(np.float64),
            model.acc_layer_1[0].bias.detach().numpy().astype(np.float64),
            model.steer_layer_1_head[0].bias.detach().numpy().astype(np.float64),
            model.steer_layer_1_tail[0].bias.detach().numpy().astype(np.float64),
            model.acc_layer_2[0].bias.detach().numpy().astype(np.float64),
            model.steer_layer_2[0].bias.detach().numpy().astype(np.float64),
            model.lstm.bias_hh_l0.detach().numpy().astype(np.float64),
            model.lstm.bias_ih_l0.detach().numpy().astype(np.float64),
            model.linear_relu_stack_1[0].bias.detach().numpy().astype(np.float64),
            model.linear_relu_stack_2[0].bias.detach().numpy().astype(np.float64),
            model.finalize.bias.detach().numpy().astype(np.float64),
        )
        self.transform.set_params_res(
            A_for_linear_reg,
            b_for_linear_reg,
            deg,
            acc_delay_step,
            steer_delay_step,
            acc_queue_size,
            steer_queue_size,
            steer_queue_size_core,
            vel_normalize,
            acc_normalize,
            steer_normalize,
        )
        self.pred = self.transform.rot_and_d_rot_error_prediction
        self.pred_only_state = self.transform.rotated_error_prediction
        self.pred_with_diff = self.transform.rot_and_d_rot_error_prediction_with_diff
        self.pred_with_memory_diff = self.transform.rot_and_d_rot_error_prediction_with_memory_diff
        self.pred_with_poly_diff = self.transform.rot_and_d_rot_error_prediction_with_poly_diff
        self.Pred = self.transform.Rotated_error_prediction
        self.test_lstm = self.transform.error_prediction
