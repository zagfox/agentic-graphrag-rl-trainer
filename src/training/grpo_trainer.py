from trl import GRPOTrainer, GRPOConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import torch
import json
from typing import Dict, List
from ..reward.composite_reward import CompositeRewardFunction
from ..utils.gpu_utils import get_gpu_memory_stats, clear_gpu_memory
import wandb


class AgenticRLTrainer:
    """
    GRPO-based RL trainer for function calling.

    GRPO (Group Relative Policy Optimization) is more modern than PPO and works
    better with Unsloth. It's simpler to use and doesn't require separate
    ref_model, value_model, or reward_model.

    Reference:
    - TRL GRPOTrainer: https://huggingface.co/docs/trl/grpo_trainer
    - Unsloth GRPO Guide: https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset,
        eval_dataset,
        reward_function: CompositeRewardFunction,
        config: DictConfig,
        experiment_config: DictConfig,
        model_config: DictConfig = None  # Accept model config to get max_seq_length
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.cfg = config
        self.exp_cfg = experiment_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.global_step = 0

        # Get max_seq_length from model config (after auto-tuning)
        # Use model config if provided, otherwise fall back to tokenizer's model_max_length
        if model_config and hasattr(model_config, 'max_sequence_length'):
            max_seq_length = model_config.max_sequence_length
        else:
            max_seq_length = getattr(tokenizer, 'model_max_length', 2048)
            if max_seq_length > 100000:  # Some tokenizers have unrealistic defaults
                max_seq_length = 2048

        # Split sequence length between prompt and completion
        # Reserve more for prompt since it contains system message + tools + query
        # Tool schemas can be long, so we allocate 75% for prompt, 25% for completion
        max_prompt_length = int(max_seq_length * 0.75)  # 75% for prompt with tool schemas
        max_completion_length = max_seq_length - max_prompt_length  # 25% for completion (function calls are short)

        print(f"GRPO Config: max_seq_length={max_seq_length}, "
              f"max_prompt_length={max_prompt_length}, "
              f"max_completion_length={max_completion_length}")

        # Get generation config from model_config if available
        temperature = getattr(model_config, 'generation', {}).get('temperature', 0.7) if model_config else 0.7
        top_p = getattr(model_config, 'generation', {}).get('top_p', 0.9) if model_config else 0.9

        # GRPO configuration
        self.grpo_config = GRPOConfig(
            learning_rate=config.ppo.learning_rate,
            per_device_train_batch_size=config.ppo.batch_size,
            gradient_accumulation_steps=config.ppo.gradient_accumulation_steps,
            num_generations=config.ppo.get("num_generations", 4),  # Number of completions per prompt
            max_prompt_length=max_prompt_length,  # Auto-tuned from model config
            max_completion_length=max_completion_length,  # Auto-tuned from model config
            max_steps=config.schedule.total_steps,
            logging_steps=config.schedule.logging_steps,
            save_steps=config.schedule.save_steps,
            max_grad_norm=config.optimization.max_grad_norm,
            warmup_steps=config.optimization.warmup_steps,
            weight_decay=config.optimization.weight_decay,
            optim=config.optimization.optimizer,
            seed=experiment_config.seed,
            report_to="wandb" if experiment_config.logging.use_wandb else "none",
            project=experiment_config.logging.wandb_project if experiment_config.logging.use_wandb else "grpo_training",
            output_dir=f"{experiment_config.output_dir}/grpo_checkpoints",
            # Generation parameters for diversity
            temperature=temperature,
            top_p=top_p,
            # Note: do_sample is not a GRPOConfig parameter - GRPO handles sampling internally
            # Disable torch.compile to avoid dynamo errors
            torch_compile=False,
        )

        # Create reward function wrapper for GRPO
        # GRPO expects reward functions with signature: (prompts, completions, **kwargs) -> List[float]
        def grpo_reward_wrapper(prompts, completions, **kwargs):
            """
            Wrapper to adapt our CompositeRewardFunction to GRPO's expected signature.

            Args:
                prompts: List of prompt strings
                completions: List of completion dicts with structure [{'content': str, 'role': 'assistant'}]
                **kwargs: Additional kwargs from dataset (e.g., ground_truth, query)

            Returns:
                List of reward scores (floats)
            """
            # Extract completion texts - handle different GRPO completion formats
            responses = []
            for comp in completions:
                try:
                    if isinstance(comp, list) and len(comp) > 0:
                        # Format: [{'content': str, 'role': 'assistant'}]
                        if isinstance(comp[0], dict):
                            responses.append(comp[0].get('content', ''))
                        else:
                            responses.append(str(comp[0]))
                    elif isinstance(comp, dict):
                        # Format: {'content': str}
                        responses.append(comp.get('content', ''))
                    elif isinstance(comp, str):
                        # Format: direct string
                        responses.append(comp)
                    else:
                        # Fallback: convert to string
                        responses.append(str(comp))
                except Exception as e:
                    self.logger.warning(f"Error parsing completion: {e}, completion type: {type(comp)}")
                    responses.append(str(comp))

            # Get ground truths from kwargs (passed from dataset)
            raw_ground_truths = kwargs.get('ground_truth', [None] * len(prompts))
            queries = kwargs.get('query', prompts)

            # Handle ground_truth deserialization (might be JSON strings)
            ground_truths = []
            for gt in raw_ground_truths:
                if isinstance(gt, str):
                    try:
                        parsed_gt = json.loads(gt)
                        ground_truths.append(parsed_gt)
                    except json.JSONDecodeError:
                        ground_truths.append(gt)  # Keep as string if not JSON
                else:
                    ground_truths.append(gt)

            # Compute rewards using our composite reward function
            rewards = self.reward_function.batch_compute_reward(
                responses,
                ground_truths,
                queries
            )

            return rewards.tolist()

        # Create GRPO trainer
        self.grpo_trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[grpo_reward_wrapper],  # GRPO accepts list of reward functions
            args=self.grpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    def train(self):
        """Main training loop using GRPO."""
        import sys
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Starting GRPO training for {self.cfg.schedule.total_steps} steps")
        print(f"Starting GRPO training for {self.cfg.schedule.total_steps} steps", flush=True)
        sys.stdout.flush()

        # SFT warm-start if configured
        if self.cfg.sft_warmstart.enabled:
            logger.info("Running SFT warm-start...")
            print("Running SFT warm-start...", flush=True)
            sys.stdout.flush()
            try:
                self.sft_warmstart()
            except Exception as e:
                logger.error(f"SFT warm-start failed: {e}")
                print(f"ERROR in SFT warm-start: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise

        # GRPO training (handled by TRL trainer)
        logger.info("Starting GRPO phase...")
        print("Starting GRPO phase...", flush=True)
        sys.stdout.flush()
        self.grpo_trainer.train()

        logger.info("Training complete!")
        print("Training complete!", flush=True)

    def sft_warmstart(self):
        """SFT warm-start before RL training."""
        from transformers import TrainingArguments, Trainer
        from pathlib import Path
        import json
        import sys
        import logging
        logger = logging.getLogger(__name__)

        logger.info("Starting SFT warm-start phase...")
        print("Starting SFT warm-start phase...", flush=True)
        sys.stdout.flush()

        # Verify dataset format
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty!")

        logger.info(f"SFT dataset size: {len(self.train_dataset)} examples")
        print(f"SFT dataset size: {len(self.train_dataset)} examples", flush=True)

        # Check first example to verify structure
        first_example = self.train_dataset[0]
        logger.info(f"Dataset columns: {list(first_example.keys())}")
        print(f"Dataset columns: {list(first_example.keys())}", flush=True)

        # Create SFT configuration
        training_args = TrainingArguments(
            output_dir=f"{self.exp_cfg.output_dir}/sft_warmstart",
            num_train_epochs=self.cfg.sft_warmstart.num_epochs,
            per_device_train_batch_size=self.cfg.sft_warmstart.batch_size,
            per_device_eval_batch_size=self.cfg.sft_warmstart.batch_size,
            learning_rate=self.cfg.sft_warmstart.learning_rate,
            logging_steps=10,
            save_steps=100,
            eval_strategy="no",
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
            gradient_accumulation_steps=self.cfg.sft_warmstart.gradient_accumulation_steps,
            warmup_steps=self.cfg.sft_warmstart.warmup_steps,
            logging_dir=f"{self.exp_cfg.output_dir}/sft_logs",
            report_to="wandb" if self.exp_cfg.logging.use_wandb else "none",
            # Disable torch compile to avoid errors
            torch_compile=False,
        )

        # Create SFT dataset (prompt + expected completion)
        def sft_format_function(example):
            """Format for supervised fine-tuning."""
            try:
                # Combine prompt with ground truth for SFT
                prompt_messages_raw = example["prompt"]  # JSON string from dataset

                # Deserialize prompt if needed
                if isinstance(prompt_messages_raw, str):
                    try:
                        prompt_messages = json.loads(prompt_messages_raw)
                    except json.JSONDecodeError:
                        # Fallback: create simple prompt
                        prompt_messages = [
                            {"role": "system", "content": "You are a function calling AI assistant."},
                            {"role": "user", "content": example.get("query", "")}
                        ]
                else:
                    prompt_messages = prompt_messages_raw  # Already a list

                # Handle both string and list formats for ground_truth
                ground_truth_raw = example["ground_truth"]
                if isinstance(ground_truth_raw, str):
                    try:
                        ground_truth_data = json.loads(ground_truth_raw)
                    except json.JSONDecodeError:
                        ground_truth_data = []
                else:
                    ground_truth_data = ground_truth_raw

                ground_truth_str = json.dumps(ground_truth_data, ensure_ascii=False)

                # Add assistant message with ground truth
                sft_messages = prompt_messages + [
                    {"role": "assistant", "content": ground_truth_str}
                ]

                # Check if tokenizer has chat template
                if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
                    # Fallback: manual formatting
                    formatted_parts = []
                    for msg in sft_messages:
                        formatted_parts.append(f"{msg['role'].upper()}: {msg['content']}")
                    formatted_text = "\n\n".join(formatted_parts)
                else:
                    formatted_text = self.tokenizer.apply_chat_template(
                        sft_messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                return {"text": formatted_text}
            except Exception as e:
                logger.error(f"Error formatting example: {e}")
                print(f"Error formatting example: {e}", flush=True)
                print(f"Example keys: {example.keys()}", flush=True)
                import traceback
                traceback.print_exc()
                # Return a dummy example to continue
                return {"text": "ERROR"}  # Mark as error but don't crash

        # Apply SFT formatting manually (datasets.map() has issues with closures)
        logger.info("Formatting dataset for SFT (manual approach)...")
        print("Formatting dataset for SFT (manual approach)...", flush=True)
        sys.stdout.flush()

        try:
            logger.info(f"Formatting {len(self.train_dataset)} examples...")
            print(f"Formatting {len(self.train_dataset)} examples...", flush=True)

            # Manually format each example
            formatted_examples = []
            for i, example in enumerate(self.train_dataset):
                if i % 20 == 0:  # Progress every 20 examples
                    logger.info(f"Formatted {i}/{len(self.train_dataset)} examples...")
                    print(f"Formatted {i}/{len(self.train_dataset)} examples...", flush=True)

                formatted = sft_format_function(example)
                formatted_examples.append(formatted)

            # Create new dataset from formatted examples
            from datasets import Dataset
            sft_dataset = Dataset.from_list(formatted_examples)

            logger.info("Formatting completed!")
            print("Formatting completed!", flush=True)
        except Exception as e:
            logger.error(f"Error during dataset formatting: {e}")
            print(f"Error during dataset formatting: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        logger.info(f"Formatted SFT dataset size: {len(sft_dataset)}")
        print(f"Formatted SFT dataset size: {len(sft_dataset)}", flush=True)

        # Verify formatted dataset
        if len(sft_dataset) == 0:
            raise ValueError("Formatted SFT dataset is empty!")

        # Check first formatted example
        first_formatted = sft_dataset[0]
        if "text" not in first_formatted:
            raise ValueError("Formatted examples missing 'text' field!")
        logger.info(f"Sample formatted text (first 200 chars): {first_formatted['text'][:200]}...")
        print(f"Sample formatted text (first 200 chars): {first_formatted['text'][:200]}...", flush=True)

        # Create custom data collator
        def sft_data_collator(examples):
            """Collate function for SFT training."""
            try:
                texts = [ex["text"] for ex in examples]
                # Use auto-tuned max_seq_length from model config
                if hasattr(self, 'grpo_config') and hasattr(self.grpo_config, 'max_prompt_length'):
                    # Use the max_seq_length that was auto-tuned
                    max_length = self.grpo_config.max_prompt_length + self.grpo_config.max_completion_length
                else:
                    max_length = getattr(self.model.config, 'max_position_embeddings', 512)

                tokenized = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                # Labels are the same as input_ids for language modeling
                tokenized["labels"] = tokenized["input_ids"].clone()
                return tokenized
            except Exception as e:
                print(f"Error in data collator: {e}")
                print(f"Example keys: {[ex.keys() for ex in examples[:2]]}")
                raise

        # Custom trainer that handles LoRA and Unsloth
        class SFTTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """
                Compute cross-entropy loss for SFT.

                Args:
                    model: The model to train
                    inputs: Input tensors dict
                    return_outputs: Whether to return model outputs
                    **kwargs: Additional arguments (e.g., num_items_in_batch from Unsloth)
                """
                try:
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                    # Shift logits and labels for next token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                    return (loss, outputs) if return_outputs else loss
                except Exception as e:
                    logger.error(f"Error in compute_loss: {e}")
                    print(f"Error in compute_loss: {e}", flush=True)
                    print(f"Input keys: {inputs.keys()}", flush=True)
                    raise

        # Initialize SFT trainer
        logger.info("Initializing SFT trainer...")
        print("Initializing SFT trainer...", flush=True)
        try:
            sft_trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=sft_dataset,
                tokenizer=self.tokenizer,
                data_collator=sft_data_collator,
            )
        except Exception as e:
            logger.error(f"Error initializing SFT trainer: {e}")
            print(f"Error initializing SFT trainer: {e}", flush=True)
            raise

        # Run SFT training
        total_steps = len(sft_dataset) // training_args.per_device_train_batch_size // training_args.gradient_accumulation_steps * self.cfg.sft_warmstart.num_epochs
        logger.info(f"Starting SFT training for {self.cfg.sft_warmstart.num_epochs} epochs (~{total_steps} steps)...")
        print(f"Starting SFT training for {self.cfg.sft_warmstart.num_epochs} epochs (~{total_steps} steps)...", flush=True)

        try:
            sft_trainer.train()
            logger.info("SFT training completed successfully!")
            print("SFT training completed successfully!", flush=True)
        except Exception as e:
            logger.error(f"Error during SFT training: {e}")
            print(f"Error during SFT training: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        # Save SFT model
        sft_save_path = Path(self.exp_cfg.output_dir) / "sft_model"
        sft_trainer.save_model(str(sft_save_path))
        logger.info(f"SFT warm-start completed! Model saved to {sft_save_path}")
        print(f"SFT warm-start completed! Model saved to {sft_save_path}", flush=True)

    def evaluate(self) -> Dict:
        """Evaluate on validation set."""
        # GRPO trainer handles evaluation automatically
        # This method can be used for custom evaluation if needed
        return {}

    def save_model(self, output_path):
        """Save final model."""
        from pathlib import Path
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.grpo_trainer.save_model(str(output_path))
        self.tokenizer.save_pretrained(output_path)
        print(f"Model saved to {output_path}")
