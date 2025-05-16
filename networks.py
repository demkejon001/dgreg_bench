import torch
import torch.nn as nn


class ResidualMLPConnection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, norm: str):
        super().__init__()
        self.l1 = nn.Linear(in_features, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        for p in self.l2.parameters():
            p.data = torch.zeros_like(p)
        if hidden_size != in_features:
            self.downsample = nn.Linear(in_features, hidden_size)
        else: 
            self.downsample = nn.Identity()
        self.activation = nn.ReLU()
        if norm == "none":
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        elif norm == "batch":
            self.norm1 = nn.BatchNorm1d(hidden_size)
            self.norm2 = nn.BatchNorm1d(hidden_size)
        elif norm == "layer":
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
        elif norm == "dropout":
            self.norm1 = nn.Dropout(.1)
            self.norm2 = nn.Dropout(.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(self.l1(self.activation(x)))
        y = self.norm2(self.l2(self.activation(y)))
        return y + self.downsample(x)


class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, hparams: dict):
        super().__init__()
        self.n_outputs = n_outputs
        hidden = hparams["mlp_width"]
        n_hidden_layers = hparams["mlp_depth"]
        norm = hparams.get("norm", "none")

        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            *[ResidualMLPConnection(hidden, hidden, norm) for _ in range(n_hidden_layers)],
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, x):
        return self.net(x)


class WorldModelFeaturizer(nn.Module):
    def __init__(self, n_inputs, n_outputs, hparams):
        super().__init__()
        hidden = hparams["cnn_hidden"]
        norm_type = hparams["norm"]

        if norm_type == "none":
            norms = [nn.Identity() for _ in range(4)]
        elif norm_type == "batch":
            norms = [nn.BatchNorm2d(hidden) for i in range(3)] + [nn.BatchNorm1d(hidden)]
        elif norm_type == "dropout":
            norms = [nn.Dropout(.1) for _ in range(4)]
        elif norm_type == "layer":
            if n_inputs[-1] == 8:
                norms = [nn.LayerNorm((6, 6)), nn.LayerNorm((4, 4)), nn.LayerNorm((2, 2)), nn.LayerNorm(hidden)] 
            else:
                norms = [nn.LayerNorm((14, 14)), nn.LayerNorm((7, 7)), nn.LayerNorm((3, 3)), nn.LayerNorm(hidden)] 
        else:
            raise ValueError(f"Unrecognized norm={norm_type}")

        if n_inputs[-1] == 8:
            self.state_net = nn.Sequential(
                nn.Conv2d(3+4, hidden, 3),  # 6x6
                norms[0],
                nn.SiLU(),
                nn.Flatten(),
                nn.Linear(hidden*36, hidden),
            )
        else:
            self.state_net = nn.Sequential(
                nn.Conv2d(3+4, hidden, 3),  # 14x14 
                norms[0],
                nn.SiLU(),
                nn.Conv2d(hidden, hidden, 2, 2),  # 7x7
                norms[1],
                nn.SiLU(),
                nn.Conv2d(hidden, hidden, 3, 2),  # 3x3
                norms[2],
                nn.SiLU(),
                nn.Conv2d(hidden, hidden, 3),  # 1x1
                nn.Flatten(),
                norms[3],
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )

        self.n_outputs = hidden

    def forward(self, state_action):
        state, action = state_action
        action = action.view(action.size(0), action.size(1), 1, 1)
        action = action.expand(-1, -1, state.size(2), state.size(3))
        state_action = torch.cat((state, action), dim=1)
        return self.state_net(state_action)


class WorldModelRegressor(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden, 3),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden, hidden, 3, 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden, hidden, 2, 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden, 3, 3),
        )
        self.reward_net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        next_state_pred = self.state_net(x.unsqueeze(-1).unsqueeze(-1))
        reward_pred = self.reward_net(x)
        return torch.cat((torch.flatten(next_state_pred, 1), reward_pred), dim=1)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    if len(input_shape) == 3:
        return WorldModelFeaturizer(input_shape, None, hparams)
    else:
        raise NotImplementedError


def Regressor(input_shape, in_features, out_features, is_nonlinear=False):
    if len(input_shape) == 3:
        return WorldModelRegressor(in_features)
        
    if is_nonlinear:
        return torch.nn.Sequential(
            ResidualMLPConnection(in_features, in_features, norm="none"),
            nn.Linear(in_features, out_features),
        )
    else:
        return nn.Linear(in_features, out_features)

