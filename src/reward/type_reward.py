from jsonschema import validate, ValidationError, Draft7Validator
from typing import Dict, Any, List
import json
from .base_reward import BaseRewardFunction


class TypeCheckingReward(BaseRewardFunction):
    """
    Reward based on parameter type correctness using JSON Schema validation.

    Reference: Function Calling Schema Validation
    https://amitness.com/posts/function-calling-schema/
    """

    def __init__(
        self,
        weight: float = 0.2,
        penalty_wrong_type: float = -0.5,
        reward_correct_type: float = 0.5
    ):
        super().__init__(weight)
        self.penalty_wrong_type = penalty_wrong_type
        self.reward_correct_type = reward_correct_type

    def _extract_schema(self, ground_truth: Dict) -> Dict:
        """Extract JSON schema from tool definition."""
        # Assuming ground_truth contains tool definitions
        # Format: {"name": "func", "arguments": {...}, "schema": {...}}
        if "schema" in ground_truth:
            return ground_truth["schema"]

        # Generate basic schema from arguments
        return self._infer_schema(ground_truth.get("arguments", {}))

    def _infer_schema(self, arguments: Dict) -> Dict:
        """Infer JSON schema from example arguments."""
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }

        for key, value in arguments.items():
            python_type = type(value)
            json_type = type_map.get(python_type, "string")
            schema["properties"][key] = {"type": json_type}
            schema["required"].append(key)

        return schema

    def compute_reward(
        self,
        predicted: str,
        ground_truth: Dict[str, Any],
        query: str = None
    ) -> float:
        """
        Validate parameter types against schema.
        """
        try:
            # Parse predicted output
            predicted_calls = json.loads(predicted)
            if not isinstance(predicted_calls, list):
                predicted_calls = [predicted_calls]

            # Get ground truth calls
            gt_calls = ground_truth if isinstance(ground_truth, list) else [ground_truth]

            if len(predicted_calls) == 0:
                return 0.0

            # Check type correctness for each call
            total_score = 0.0
            for pred_call in predicted_calls:
                # Find matching ground truth call
                matching_gt = next(
                    (gt for gt in gt_calls if gt.get("name") == pred_call.get("name")),
                    None
                )

                if matching_gt is None:
                    continue

                # Validate types
                schema = self._extract_schema(matching_gt)
                try:
                    validate(instance=pred_call.get("arguments", {}), schema=schema)
                    total_score += self.reward_correct_type
                except ValidationError as e:
                    total_score += self.penalty_wrong_type

            # Average over all calls
            return total_score / len(predicted_calls) if predicted_calls else 0.0

        except (json.JSONDecodeError, TypeError, AttributeError):
            return self.penalty_wrong_type
