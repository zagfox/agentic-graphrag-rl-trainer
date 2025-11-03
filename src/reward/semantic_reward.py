import json
from typing import Dict, Any, Set
from .base_reward import BaseRewardFunction


class SemanticCorrectnessReward(BaseRewardFunction):
    """
    Semantic correctness using AST Score from BFCL.

    AST Score breakdown:
    - Function name matching: 40%
    - Parameter name matching: 30%
    - Parameter value matching: 30%

    Reference: https://arxiv.org/html/2502.00032v1
    """

    def __init__(
        self,
        weight: float = 0.5,
        func_name_weight: float = 0.4,
        param_name_weight: float = 0.3,
        param_value_weight: float = 0.3
    ):
        super().__init__(weight)
        self.func_name_weight = func_name_weight
        self.param_name_weight = param_name_weight
        self.param_value_weight = param_value_weight

        # Validate weights sum to 1.0
        total = func_name_weight + param_name_weight + param_value_weight
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"

    def compute_reward(
        self,
        predicted: str,
        ground_truth: Dict[str, Any],
        query: str = None
    ) -> float:
        """
        Compute AST score with support for multi-call function calling.

        Enhanced algorithm:
        1. Handle multiple predicted and ground truth calls
        2. Use optimal matching (Hungarian algorithm) for best pairing
        3. Score individual calls and average appropriately

        Returns 0.0 if function name is wrong (as per BFCL spec).
        """
        try:
            # Parse predicted
            predicted_calls = json.loads(predicted)
            if not isinstance(predicted_calls, list):
                predicted_calls = [predicted_calls]

            # Get ground truth
            gt_calls = ground_truth if isinstance(ground_truth, list) else [ground_truth]

            if len(predicted_calls) == 0 or len(gt_calls) == 0:
                return 0.0

            # Use optimal matching for multi-call scenarios
            if len(predicted_calls) > 1 or len(gt_calls) > 1:
                return self._multi_call_score(predicted_calls, gt_calls)

            # Single call scoring (existing logic)
            return self._single_call_score(predicted_calls[0], gt_calls[0])

        except (json.JSONDecodeError, TypeError, AttributeError, KeyError):
            return 0.0

    def _single_call_score(self, pred_call: Dict, gt_call: Dict) -> float:
        """Score single function call."""
        # 1. Function name matching (40%)
        pred_name = pred_call.get("name", "")
        gt_name = gt_call.get("name", "")

        if pred_name != gt_name:
            # If function name is wrong, return 0 (BFCL spec)
            return 0.0

        score = self.func_name_weight  # Function name is correct

        # 2. Parameter name matching (30%)
        pred_args = pred_call.get("arguments", {})
        gt_args = gt_call.get("arguments", {})

        pred_param_names = set(pred_args.keys())
        gt_param_names = set(gt_args.keys())

        if len(gt_param_names) > 0:
            param_name_score = len(pred_param_names & gt_param_names) / len(gt_param_names)
            score += self.param_name_weight * param_name_score

        # 3. Parameter value matching (30%)
        if len(gt_args) > 0:
            matching_values = sum(
                1 for key in gt_args
                if pred_args.get(key) == gt_args[key]
            )
            param_value_score = matching_values / len(gt_args)
            score += self.param_value_weight * param_value_score

        return score

    def _multi_call_score(self, predicted_calls: list, gt_calls: list) -> float:
        """
        Score multi-call function calling using optimal matching.

        Uses Hungarian algorithm to find optimal pairing between predicted and ground truth calls.
        """
        try:
            from scipy.optimize import linear_sum_assignment
            import numpy as np
        except ImportError:
            # Fallback to simple sequential matching if scipy not available
            return self._simple_multi_call_score(predicted_calls, gt_calls)

        # Create cost matrix (negative scores since we want to maximize)
        cost_matrix = np.zeros((len(predicted_calls), len(gt_calls)))

        for i, pred_call in enumerate(predicted_calls):
            for j, gt_call in enumerate(gt_calls):
                # Compute similarity score for this pairing
                score = self._compute_call_similarity(pred_call, gt_call)
                cost_matrix[i, j] = -score  # Negative for minimization

        # Find optimal assignment
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            # Compute total score for optimal matching
            total_score = 0.0

            for i, j in zip(row_indices, col_indices):
                similarity = -cost_matrix[i, j]
                total_score += similarity

            # Normalize by number of expected calls
            average_score = total_score / len(gt_calls) if len(gt_calls) > 0 else 0.0

            # Penalty for incorrect number of calls
            call_count_penalty = 0.0
            if len(predicted_calls) != len(gt_calls):
                # Small penalty for wrong number of calls
                call_count_penalty = -0.1 * abs(len(predicted_calls) - len(gt_calls))

            return max(0.0, average_score + call_count_penalty)

        return 0.0

    def _simple_multi_call_score(self, predicted_calls: list, gt_calls: list) -> float:
        """Simple sequential matching fallback when scipy is not available."""
        total_score = 0.0
        max_calls = min(len(predicted_calls), len(gt_calls))

        for i in range(max_calls):
            total_score += self._single_call_score(predicted_calls[i], gt_calls[i])

        # Average and apply penalty for wrong count
        average_score = total_score / len(gt_calls) if len(gt_calls) > 0 else 0.0
        call_count_penalty = -0.1 * abs(len(predicted_calls) - len(gt_calls))

        return max(0.0, average_score + call_count_penalty)

    def _compute_call_similarity(self, pred_call: Dict, gt_call: Dict) -> float:
        """Compute similarity score between two function calls."""
        similarity = 0.0

        # Function name matching (40% weight)
        if pred_call.get("name") == gt_call.get("name"):
            similarity += 0.4
        else:
            return 0.0  # Wrong function name = zero similarity

        # Parameter name matching (30% weight)
        pred_args = set(pred_call.get("arguments", {}).keys())
        gt_args = set(gt_call.get("arguments", {}).keys())

        if len(gt_args) > 0:
            param_name_similarity = len(pred_args & gt_args) / len(gt_args)
            similarity += 0.3 * param_name_similarity

        # Parameter value matching (30% weight)
        pred_params = pred_call.get("arguments", {})
        gt_params = gt_call.get("arguments", {})

        if len(gt_params) > 0:
            matching_values = sum(
                1 for key in gt_params
                if pred_params.get(key) == gt_params[key]
            )
            param_value_similarity = matching_values / len(gt_params)
            similarity += 0.3 * param_value_similarity

        return similarity
