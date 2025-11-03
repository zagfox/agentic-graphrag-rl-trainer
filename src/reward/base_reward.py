from abc import ABC, abstractmethod
from typing import Dict, List, Any
import torch


class BaseRewardFunction(ABC):
    """Abstract base class for reward functions."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @abstractmethod
    def compute_reward(
        self,
        predicted: str,
        ground_truth: Dict[str, Any],
        query: str = None
    ) -> float:
        """
        Compute reward for a single prediction.

        Args:
            predicted: Model's generated response
            ground_truth: Expected correct answer
            query: Original query (optional, for context)

        Returns:
            reward: Float reward value
        """
        pass

    def batch_compute_reward(
        self,
        predicted_list: List[str],
        ground_truth_list: List[Dict],
        query_list: List[str] = None
    ) -> torch.Tensor:
        """Compute rewards for a batch."""
        if query_list is None:
            query_list = [None] * len(predicted_list)

        rewards = [
            self.compute_reward(pred, gt, query)
            for pred, gt, query in zip(predicted_list, ground_truth_list, query_list)
        ]

        return torch.tensor(rewards, dtype=torch.float32) * self.weight
