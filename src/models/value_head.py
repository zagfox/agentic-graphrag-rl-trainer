import torch
import torch.nn as nn
from transformers import PreTrainedModel


class ValueHead(nn.Module):
    """
    Value head for PPO critic.

    Takes hidden states from LLM and outputs scalar value.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            values: [batch_size, seq_len, 1]
        """
        return self.value_head(hidden_states)


class ModelWithValueHead(nn.Module):
    """Wrapper that adds value head to language model."""

    def __init__(self, base_model: PreTrainedModel):
        super().__init__()
        self.base_model = base_model
        self.value_head = ValueHead(base_model.config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_value: bool = False,
        **kwargs
    ):
        """Forward pass with optional value head."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        if return_value:
            # Get last hidden state and compute value
            hidden_states = outputs.hidden_states[-1]
            values = self.value_head(hidden_states)
            return outputs, values

        return outputs
