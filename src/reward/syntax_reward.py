import ast
import json
from typing import Dict, Any
from .base_reward import BaseRewardFunction


class SyntaxValidationReward(BaseRewardFunction):
    """
    Reward based on JSON syntax validation using AST parsing.

    Reference: BFCL AST Matching
    https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html
    """

    def __init__(
        self,
        weight: float = 0.3,
        penalty_malformed: float = -1.0,
        reward_valid: float = 1.0
    ):
        super().__init__(weight)
        self.penalty_malformed = penalty_malformed
        self.reward_valid = reward_valid

    def compute_reward(
        self,
        predicted: str,
        ground_truth: Dict[str, Any],
        query: str = None
    ) -> float:
        """
        Validate JSON syntax using AST parsing.

        Steps:
        1. Try to parse as JSON
        2. If fails, try ast.literal_eval (handles Python dict syntax)
        3. Validate structure (must be list of tool calls)
        """
        try:
            # Try JSON parsing first
            parsed = json.loads(predicted)

            # Validate structure
            if not isinstance(parsed, list):
                if isinstance(parsed, dict):
                    parsed = [parsed]  # Single tool call
                else:
                    return self.penalty_malformed

            # Validate each tool call has required fields
            for call in parsed:
                if not isinstance(call, dict):
                    return self.penalty_malformed

                # Must have 'name' field
                if 'name' not in call:
                    return self.penalty_malformed

                # 'arguments' should be dict
                if 'arguments' in call and not isinstance(call['arguments'], dict):
                    return self.penalty_malformed

            return self.reward_valid

        except json.JSONDecodeError:
            # Try Python literal eval as fallback
            try:
                parsed = ast.literal_eval(predicted)

                # Same validation as above
                if not isinstance(parsed, list):
                    if isinstance(parsed, dict):
                        parsed = [parsed]
                    else:
                        return self.penalty_malformed

                for call in parsed:
                    if not isinstance(call, dict):
                        return self.penalty_malformed
                    if 'name' not in call:
                        return self.penalty_malformed
                    if 'arguments' in call and not isinstance(call['arguments'], dict):
                        return self.penalty_malformed

                return self.reward_valid

            except (ValueError, SyntaxError):
                # Cannot parse at all
                return self.penalty_malformed
