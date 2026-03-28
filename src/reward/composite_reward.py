from typing import Dict, Any, List
import torch
from .base_reward import BaseRewardFunction
from .syntax_reward import SyntaxValidationReward
from .type_reward import TypeCheckingReward
from .semantic_reward import SemanticCorrectnessReward
from omegaconf import DictConfig


class CompositeRewardFunction(BaseRewardFunction):
    """
    Composite reward combining multiple reward components.

    Pipeline:
    1. Syntax validation (must pass to get further rewards)
    2. Type checking
    3. Semantic correctness

    Final reward = weighted sum of all components.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(weight=1.0)

        # Initialize component rewards
        self.syntax_reward = SyntaxValidationReward(
            weight=cfg.reward.components.syntax.weight,
            penalty_malformed=cfg.reward.components.syntax.penalty_malformed,
            reward_valid=cfg.reward.components.syntax.reward_valid
        ) if cfg.reward.components.syntax.enabled else None

        self.type_reward = TypeCheckingReward(
            weight=cfg.reward.components.type_checking.weight,
            penalty_wrong_type=cfg.reward.components.type_checking.penalty_wrong_type,
            reward_correct_type=cfg.reward.components.type_checking.reward_correct_type
        ) if cfg.reward.components.type_checking.enabled else None

        self.semantic_reward = SemanticCorrectnessReward(
            weight=cfg.reward.components.semantic.weight,
            func_name_weight=cfg.reward.components.semantic.function_name.weight,
            param_name_weight=cfg.reward.components.semantic.parameter_names.weight,
            param_value_weight=cfg.reward.components.semantic.parameter_values.weight
        ) if cfg.reward.components.semantic.enabled else None

        # Normalization config
        self.normalization_method = cfg.reward.normalization.method
        self.clip_range = cfg.reward.normalization.clip_range

        # Track statistics for normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = []

    def compute_reward(
        self,
        predicted: str,
        ground_truth: Dict[str, Any],
        query: str = None
    ) -> float:
        """Compute composite reward."""
        total_reward = 0.0

        # 1. Syntax validation (gating)
        if self.syntax_reward is not None:
            syntax_score = self.syntax_reward.compute_reward(predicted, ground_truth, query)
            total_reward += syntax_score

            # If syntax is invalid, don't compute other rewards
            if syntax_score < 0:
                return total_reward

        # 2. Type checking
        if self.type_reward is not None:
            type_score = self.type_reward.compute_reward(predicted, ground_truth, query)
            total_reward += type_score

        # 3. Semantic correctness
        if self.semantic_reward is not None:
            semantic_score = self.semantic_reward.compute_reward(predicted, ground_truth, query)
            total_reward += semantic_score

        return total_reward

    def batch_compute_reward(
        self,
        predicted_list: List[str],
        ground_truth_list: List[Dict],
        query_list: List[str] = None
    ) -> torch.Tensor:
        """Compute rewards for batch with progressive reward shaping."""
        import logging
        import json
        logger = logging.getLogger(__name__)

        # Apply progressive reward shaping BEFORE the composite reward calculation
        shaped_rewards = []
        shaped_predictions = []

        logger.debug(f"🔍 DEBUG Progressive Reward Shaping:")
        logger.info(f"  Batch size: {len(predicted_list)}")

        for i, (pred, gt, query) in enumerate(zip(predicted_list, ground_truth_list, query_list)):
            shaped_reward = -1.0  # Base penalty

            # Progressive reward shaping for debugging
            pred_str = str(pred).strip()

            # Step 1: Reward any attempt at structured output
            if '{' in pred_str and '}' in pred_str:
                shaped_reward += 0.5  # +0.5 for attempting JSON
                logger.debug(f"    Sample {i}: JSON attempt detected (+0.5)")

            # Step 2: Reward valid JSON structure (robust extraction)
            try:
                if isinstance(pred, str):
                    # Try to extract JSON from text with prefixes
                    parsed_pred = self._extract_json_from_text(pred)
                else:
                    parsed_pred = pred

                shaped_reward += 0.3  # +0.3 for valid JSON
                logger.debug(f"    Sample {i}: Valid JSON (+0.3)")

                # Step 3: Reward function call structure
                if isinstance(parsed_pred, list) and len(parsed_pred) > 0:
                    first_item = parsed_pred[0]
                    if isinstance(first_item, dict) and 'name' in first_item:
                        shaped_reward += 0.2  # +0.2 for function name field
                        logger.debug(f"    Sample {i}: Function name field (+0.2)")

                    if isinstance(first_item, dict) and 'arguments' in first_item:
                        shaped_reward += 0.2  # +0.2 for arguments field
                        logger.debug(f"    Sample {i}: Arguments field (+0.2)")

                # Step 4: Use original reward if JSON structure is valid
                original_reward = super().batch_compute_reward([pred], [gt], [query])[0].item()
                if original_reward > -1.0:
                    # Blend progressive and original reward
                    shaped_reward = max(shaped_reward, original_reward)
                    logger.debug(f"    Sample {i}: Good semantic match, keeping original reward: {original_reward:.3f}")

            except (json.JSONDecodeError, Exception) as e:
                logger.debug(f"    Sample {i}: Invalid JSON, keeping base reward: {shaped_reward:.3f}")

            shaped_rewards.append(shaped_reward)

            # Update prediction for semantic reward (use original if valid JSON, otherwise keep as is)
            try:
                if isinstance(pred, str) and json.loads(pred):
                    shaped_predictions.append(pred)
                else:
                    shaped_predictions.append(pred)
            except:
                shaped_predictions.append(pred)

            # Log sample details
            logger.info(f"    Sample {i}:")
            logger.info(f"      Predicted: {pred_str[:100]}...")
            logger.info(f"      Ground truth: {gt}")
            logger.info(f"      Progressive shaped reward: {shaped_reward:.3f}")

        # Convert to tensor
        shaped_rewards_tensor = torch.tensor(shaped_rewards, dtype=torch.float32)

        logger.debug(f"  Shaped rewards: {shaped_rewards}")
        logger.debug(f"  Shaped reward mean: {shaped_rewards_tensor.mean().item():.4f}")
        logger.debug(f"  Shaped reward std: {shaped_rewards_tensor.std().item():.4f}")

        # Now apply original composite reward logic with shaped rewards
        # Use shaped_predictions for semantic matching if they have valid JSON
        rewards = shaped_rewards_tensor

        # Update statistics
        self.reward_history.extend(rewards.tolist())
        if len(self.reward_history) > 1000:  # Keep last 1000
            self.reward_history = self.reward_history[-1000:]

        self.reward_mean = sum(self.reward_history) / len(self.reward_history)
        if len(self.reward_history) > 1:
            variance = sum((r - self.reward_mean) ** 2 for r in self.reward_history) / len(self.reward_history)
            self.reward_std = variance ** 0.5

        #logger.debug(f"  Reward history stats: mean={self.reward_mean:.4f}, std={self.reward_std:.4f}")

        # Normalize
        if self.normalization_method == "z_score":
            if self.reward_std > 1e-8:  # Avoid division by zero
                rewards = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
                logger.debug(f"  Normalized rewards (z-score): {rewards.tolist()}")
            else:
                logger.debug(f"  Skipping z-score normalization (std too small: {self.reward_std:.6f})")
        elif self.normalization_method == "none":
            logger.debug(f"  No normalization applied (method=none)")
        else:
            logger.warning(f"  Unknown normalization method: {self.normalization_method}, skipping normalization")

        # Clip
        if self.clip_range is not None:
            rewards = torch.clamp(rewards, self.clip_range[0], self.clip_range[1])
            logger.debug(f"  Clipped rewards: {rewards.tolist()}")

        return rewards

    def _extract_json_from_text(self, text: str):
        """Extract JSON from text that may have prefixes or noise."""
        import re
        import json

        # Strategy 1: Try direct JSON parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find JSON array pattern [...]
        json_array_match = re.search(r'\[.*?\]', text, re.DOTALL)
        if json_array_match:
            try:
                return json.loads(json_array_match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON object pattern {...}
        json_object_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_object_match:
            try:
                return [json.loads(json_object_match.group())]  # Wrap in array for consistency
            except json.JSONDecodeError:
                pass

        # Strategy 4: Find multiple JSON objects
        json_objects = re.findall(r'\{[^{}]*\}', text)
        if json_objects:
            try:
                return [json.loads(obj) for obj in json_objects]
            except json.JSONDecodeError:
                pass

        # If all strategies fail, raise original error
        raise json.JSONDecodeError("Could not extract valid JSON from text", text, 0)
