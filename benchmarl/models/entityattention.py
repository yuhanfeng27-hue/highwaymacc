"""Ego-query attention for flattened vehicle observations.

Each entity is [entity_type, x, y, vx, vy]; entity_type=0 marks padding.
Only entities inside each agent's own observation are aggregated.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torchrl.modules import MultiAgentMLP

from .common import Model, ModelConfig


class VehicleAttentionEncoder(nn.Module):
    def __init__(self, entity_features=5, embed_dim=32, num_heads=4):
        super().__init__()
        if entity_features < 1 or embed_dim < 1 or num_heads < 1 or embed_dim % num_heads:
            raise ValueError("Positive dimensions required; embed_dim must be divisible by num_heads")
        self.entity_features = entity_features
        self.embed_dim = embed_dim
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_features, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, observation):
        # Fold all leading dimensions, including the agent axis, into a batch.
        leading_shape = observation.shape[:-1]
        entities = observation.reshape(-1, observation.shape[-1] // self.entity_features,
                                       self.entity_features)
        padding = entities[..., 0] == 0
        # The first entity is always ego: keep it valid even for synthetic all-zero inputs.
        padding = torch.cat((torch.zeros_like(padding[:, :1]), padding[:, 1:]), dim=1)
        # Erase padded payloads as well as masking attention keys/values.
        entities = entities.masked_fill(padding.unsqueeze(-1), 0)
        encoded = self.entity_encoder(entities)
        ego = encoded[:, :1]
        context, _ = self.attention(ego, encoded, encoded,
                                    key_padding_mask=padding, need_weights=False)
        # Preserve ego information through a residual without expanding the 32-D output.
        return self.norm(ego + context).squeeze(1).reshape(*leading_shape, self.embed_dim)


class EntityAttention(Model):
    """Vehicle encoder followed by a decentralized policy MLP.

    This model expects a single flattened vehicle-observation key. The centralized
    critic should continue to use the ordinary MLP model.
    """

    def __init__(self, entity_features=5, embed_dim=32, num_heads=4,
                 num_cells=(256, 256), **kwargs):
        self.entity_features = entity_features
        super().__init__(**kwargs)
        self.encoders = nn.ModuleList([
            VehicleAttentionEncoder(entity_features, embed_dim, num_heads)
            for _ in range(1 if self.share_params else self.n_agents)
        ]).to(self.device)
        self.policy_head = MultiAgentMLP(
            n_agent_inputs=embed_dim, n_agent_outputs=self.output_leaf_spec.shape[-1],
            n_agents=self.n_agents, centralised=False, share_params=self.share_params,
            num_cells=num_cells, activation_class=nn.Tanh, device=self.device,
        )

    def _perform_checks(self):
        super()._perform_checks()
        if self.centralised or not self.input_has_agent_dim:
            raise ValueError("EntityAttention is a decentralized actor; use MLP for the critic")
        if len(self.in_keys) != 1:
            raise ValueError("EntityAttention expects one flattened vehicle-observation key")
        shape = self.input_leaf_spec.shape
        if (self.entity_features < 1 or len(shape) != 2 or shape[-2] != self.n_agents
                or shape[-1] < self.entity_features or shape[-1] % self.entity_features):
            raise ValueError("Observation must have shape (agents, entities * entity_features)")
        if len(self.output_leaf_spec.shape) != 2 or self.output_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError("Actor output must have shape (agents, outputs)")

    def _forward(self, tensordict):
        observation = tensordict.get(self.in_key)
        if self.share_params:
            features = self.encoders[0](observation)
        else:
            features = torch.stack([
                encoder(observation[..., i, :])
                for i, encoder in enumerate(self.encoders)
            ], dim=-2)
        tensordict.set(self.out_key, self.policy_head(features))
        return tensordict


@dataclass
class EntityAttentionConfig(ModelConfig):
    entity_features: int = 5
    embed_dim: int = 32
    num_heads: int = 4
    num_cells: Sequence[int] = (256, 256)

    @staticmethod
    def associated_class():
        return EntityAttention
