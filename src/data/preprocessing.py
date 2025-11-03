from transformers import PreTrainedTokenizer
from typing import Dict, List
from dataclasses import dataclass
import torch


@dataclass
class DataCollatorForFunctionCalling:
    """
    Custom data collator for function calling with RL.

    Unlike supervised learning, RL needs:
    - query: the input prompt (system + user messages)
    - response: model-generated (not from dataset)
    - ground_truth: for reward calculation
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples.

        For RL training:
        - Only tokenize query (system + user)
        - Keep ground_truth as-is for reward calculation
        - Model will generate response during training
        """
        queries = []
        ground_truths = []

        for feature in features:
            messages = feature["messages"]

            # Build query from system + user messages
            query_messages = [m for m in messages if m["role"] != "assistant"]
            query_text = self.tokenizer.apply_chat_template(
                query_messages,
                tokenize=False,
                add_generation_prompt=True  # Add assistant prompt
            )
            queries.append(query_text)
            ground_truths.append(feature["ground_truth"])

        # Tokenize queries
        tokenized = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "query": queries,  # Keep original text for logging
            "ground_truth": ground_truths,  # For reward function
        }
