import json
import ast
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    exact_match: float
    ast_score: float
    syntax_accuracy: float
    type_accuracy: float
    function_name_accuracy: float
    num_samples: int


class FunctionCallingEvaluator:
    """Evaluator for function calling metrics."""

    def __init__(self,
                 strict_mode: bool = False,
                 cache_schemas: bool = True,
                 log_failures: bool = False):
        """
        Initialize the function calling evaluator.

        Args:
            strict_mode: If True, require exact matches. If False, allow partial matches.
            cache_schemas: If True, cache inferred schemas for performance.
            log_failures: If True, log detailed information about evaluation failures.
        """
        self.strict_mode = strict_mode
        self.cache_schemas = cache_schemas
        self.log_failures = log_failures

        # Schema cache for performance optimization
        self._schema_cache = {} if cache_schemas else None

        # Statistics tracking
        self.evaluation_stats = {
            "total_evaluations": 0,
            "syntax_errors": 0,
            "type_errors": 0,
            "semantic_errors": 0,
            "exact_matches": 0
        }

        # Initialize JSON schema validator
        self._init_schema_validator()

    def _init_schema_validator(self):
        """Initialize JSON schema validator with custom format checkers."""
        try:
            from jsonschema import Draft7Validator
            self.schema_validator = Draft7Validator
        except ImportError:
            print("Warning: jsonschema not installed. Type checking will use basic validation.")
            self.schema_validator = None

    def exact_match(self, predicted: str, ground_truth: str) -> bool:
        """Exact string match."""
        return predicted.strip() == ground_truth.strip()

    def syntax_valid(self, predicted: str) -> bool:
        """Check if output is syntactically valid JSON."""
        try:
            json.loads(predicted)
            return True
        except json.JSONDecodeError:
            try:
                ast.literal_eval(predicted)
                return True
            except (ValueError, SyntaxError):
                return False

    def ast_score(self, predicted: str, ground_truth: Dict) -> float:
        """
        Compute AST score (BFCL metric).

        Returns:
            Score between 0.0 and 1.0
        """
        try:
            pred_calls = json.loads(predicted)
            if not isinstance(pred_calls, list):
                pred_calls = [pred_calls]

            gt_calls = ground_truth if isinstance(ground_truth, list) else [ground_truth]

            if len(pred_calls) == 0 or len(gt_calls) == 0:
                return 0.0

            # Score first call
            pred_call = pred_calls[0]
            gt_call = gt_calls[0]

            score = 0.0

            # Function name (40%)
            if pred_call.get("name") == gt_call.get("name"):
                score += 0.4
            else:
                return 0.0  # Wrong function = 0 score

            # Parameters (60%)
            pred_args = pred_call.get("arguments", {})
            gt_args = gt_call.get("arguments", {})

            # Param names (30%)
            if len(gt_args) > 0:
                matching_names = len(set(pred_args.keys()) & set(gt_args.keys()))
                score += 0.3 * (matching_names / len(gt_args))

            # Param values (30%)
            if len(gt_args) > 0:
                matching_values = sum(
                    1 for k in gt_args if pred_args.get(k) == gt_args[k]
                )
                score += 0.3 * (matching_values / len(gt_args))

            return score

        except (json.JSONDecodeError, TypeError, KeyError):
            return 0.0

    def function_name_correct(self, predicted: str, ground_truth: Dict) -> bool:
        """Check if function name is correct."""
        try:
            pred_calls = json.loads(predicted)
            if not isinstance(pred_calls, list):
                pred_calls = [pred_calls]

            gt_calls = ground_truth if isinstance(ground_truth, list) else [ground_truth]

            if len(pred_calls) == 0 or len(gt_calls) == 0:
                return False

            return pred_calls[0].get("name") == gt_calls[0].get("name")

        except (json.JSONDecodeError, TypeError, KeyError):
            return False

    def type_check(self, predicted: str, ground_truth: Dict) -> bool:
        """Check if parameter types match the expected schema."""
        try:
            predicted_calls = json.loads(predicted)
            if not isinstance(predicted_calls, list):
                predicted_calls = [predicted_calls]

            gt_calls = ground_truth if isinstance(ground_truth, list) else [ground_truth]

            if len(predicted_calls) == 0 or len(gt_calls) == 0:
                return False

            # Check first call (can be extended for multi-call)
            pred_call = predicted_calls[0]
            gt_call = gt_calls[0]

            # Must have same function name
            if pred_call.get("name") != gt_call.get("name"):
                return False

            # Get or infer schema
            schema = self._extract_or_infer_schema(gt_call)
            if not schema:
                return True  # No schema to validate against

            # Validate types
            pred_args = pred_call.get("arguments", {})
            gt_args = gt_call.get("arguments", {})

            return self._validate_parameter_types(pred_args, gt_args, schema)

        except (json.JSONDecodeError, TypeError, KeyError):
            return False

    def _extract_or_infer_schema(self, gt_call: Dict) -> Dict:
        """Extract schema from ground truth or infer from example."""
        # Check cache first
        cache_key = str(gt_call.get("name", ""))
        if self._schema_cache is not None and cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        # Check if schema is provided
        if "schema" in gt_call:
            schema = gt_call["schema"]
        # Infer from function definition if available
        elif "function_def" in gt_call:
            schema = gt_call["function_def"].get("parameters", {})
        # Infer basic schema from ground truth arguments
        else:
            schema = self._infer_schema_from_args(gt_call.get("arguments", {}))

        # Cache the schema
        if self._schema_cache is not None:
            self._schema_cache[cache_key] = schema

        return schema

    def _infer_schema_from_args(self, args: Dict) -> Dict:
        """Infer JSON schema from example arguments."""
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }

        for key, value in args.items():
            python_type = type(value)
            json_type = type_mapping.get(python_type, "string")

            schema["properties"][key] = {"type": json_type}
            if key in args:  # Required if present in ground truth
                schema["required"].append(key)

        return schema

    def _validate_parameter_types(self, pred_args: Dict, gt_args: Dict, schema: Dict) -> bool:
        """Validate that predicted arguments match expected types."""
        # Try using jsonschema if available
        if self.schema_validator is not None:
            try:
                from jsonschema import validate, ValidationError

                # First check if all required parameters are present
                required_params = schema.get("required", [])
                for param in required_params:
                    if param not in pred_args:
                        return False

                # Validate types against schema
                validate(instance=pred_args, schema=schema)
                return True

            except ValidationError:
                return False
            except Exception:
                # If schema validation fails, fall back to basic type checking
                pass

        # Fallback to basic type checking
        return self._basic_type_check(pred_args, gt_args)

    def _basic_type_check(self, pred_args: Dict, gt_args: Dict) -> bool:
        """Basic type checking as fallback."""
        if len(pred_args) != len(gt_args):
            return False

        for key, pred_value in pred_args.items():
            if key not in gt_args:
                return False

            gt_value = gt_args[key]

            # Basic type compatibility
            if type(pred_value) != type(gt_value):
                # Allow numeric type flexibility
                if isinstance(pred_value, (int, float)) and isinstance(gt_value, (int, float)):
                    continue
                return False

        return True

    def get_evaluation_report(self) -> Dict:
        """Get comprehensive evaluation statistics."""
        total = self.evaluation_stats["total_evaluations"]
        if total == 0:
            return {"message": "No evaluations performed yet"}

        return {
            "total_evaluations": total,
            "success_rate": (total - sum([
                self.evaluation_stats["syntax_errors"],
                self.evaluation_stats["type_errors"],
                self.evaluation_stats["semantic_errors"]
            ])) / total,
            "error_breakdown": {
                "syntax_errors": self.evaluation_stats["syntax_errors"],
                "type_errors": self.evaluation_stats["type_errors"],
                "semantic_errors": self.evaluation_stats["semantic_errors"]
            },
            "exact_matches": self.evaluation_stats["exact_matches"],
            "exact_match_rate": self.evaluation_stats["exact_matches"] / total,
            "cache_stats": {
                "schemas_cached": len(self._schema_cache) if self._schema_cache else 0,
                "cache_enabled": self.cache_schemas
            }
        }

    def evaluate_batch(
        self,
        predictions: List[str],
        ground_truths: List[Dict]
    ) -> EvaluationResult:
        """Evaluate a batch of predictions."""
        assert len(predictions) == len(ground_truths)

        exact_matches = 0
        ast_scores = []
        syntax_valid_count = 0
        type_valid_count = 0
        func_name_correct_count = 0

        for pred, gt in zip(predictions, ground_truths):
            self.evaluation_stats["total_evaluations"] += 1

            # Exact match
            gt_str = json.dumps(gt, ensure_ascii=False)
            if self.exact_match(pred, gt_str):
                exact_matches += 1
                self.evaluation_stats["exact_matches"] += 1

            # Syntax
            if self.syntax_valid(pred):
                syntax_valid_count += 1
            else:
                self.evaluation_stats["syntax_errors"] += 1

            # AST score
            ast_scores.append(self.ast_score(pred, gt))

            # Function name
            if self.function_name_correct(pred, gt):
                func_name_correct_count += 1

            # Type accuracy (NOW IMPLEMENTED)
            if self.type_check(pred, gt):
                type_valid_count += 1
            else:
                self.evaluation_stats["type_errors"] += 1

        return EvaluationResult(
            exact_match=exact_matches / len(predictions),
            ast_score=sum(ast_scores) / len(ast_scores),
            syntax_accuracy=syntax_valid_count / len(predictions),
            type_accuracy=type_valid_count / len(predictions),  # Now properly computed
            function_name_accuracy=func_name_correct_count / len(predictions),
            num_samples=len(predictions)
        )
