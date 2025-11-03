from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import torch
from typing import Dict, List
from ..reward.composite_reward import CompositeRewardFunction
from ..utils.gpu_utils import get_gpu_memory_stats, clear_gpu_memory
import wandb


class AgenticRLTrainer:
    """
    PPO-based RL trainer for function calling.

    Reference:
    - TRL PPOTrainer: https://huggingface.co/docs/trl/ppo_trainer
    - Function calling RL: https://arxiv.org/html/2508.05118v1
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset,
        eval_dataset,
        reward_function: CompositeRewardFunction,
        config: DictConfig,
        experiment_config: DictConfig
    ):
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.cfg = config
        self.exp_cfg = experiment_config

        # Wrap model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        # Create reference model (frozen copy for KL penalty)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # PPO configuration
        self.ppo_config = PPOConfig(
            learning_rate=config.ppo.learning_rate,
            batch_size=config.ppo.batch_size,
            mini_batch_size=config.ppo.mini_batch_size,
            gradient_accumulation_steps=config.ppo.gradient_accumulation_steps,
            num_ppo_epochs=config.ppo.num_ppo_epochs,
            kl_coef=config.ppo.ppo_params.init_kl_coef,
            gamma=config.ppo.ppo_params.gamma,
            lam=config.ppo.ppo_params.lam,
            cliprange=config.ppo.ppo_params.cliprange,
            cliprange_value=config.ppo.ppo_params.cliprange_value,
            vf_coef=config.ppo.ppo_params.vf_coef,
            max_grad_norm=config.optimization.max_grad_norm,
            seed=experiment_config.seed,
            report_to="wandb" if experiment_config.logging.use_wandb else None,
            # Wandb project name
            project=experiment_config.logging.wandb_project if experiment_config.logging.use_wandb else "ppo_training",
        )

        # Create PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            data_collator=None,  # We'll handle batching manually
        )

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.global_step = 0

    def generate_responses(
        self,
        queries: List[str],
        batch_size: int = 4
    ) -> List[str]:
        """Generate responses for queries using current policy."""
        self.model.eval()

        responses = []
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i+batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_queries,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)

                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Decode (remove input prompt)
                for j, output in enumerate(outputs):
                    input_len = inputs["input_ids"][j].shape[0]
                    response = self.tokenizer.decode(
                        output[input_len:],
                        skip_special_tokens=True
                    )
                    responses.append(response)

        self.model.train()
        return responses

    def train_step(self, batch: Dict) -> Dict:
        """Single PPO training step."""
        queries = batch["query"]
        ground_truths = batch["ground_truth"]

        # Generate responses
        responses = self.generate_responses(queries)

        # Compute rewards
        rewards = self.reward_function.batch_compute_reward(
            responses,
            ground_truths,
            queries
        )

        # Tokenize query and response for PPO
        query_tensors = []
        response_tensors = []

        for query, response in zip(queries, responses):
            # Query
            query_tokens = self.tokenizer.encode(query, return_tensors="pt")[0]
            query_tensors.append(query_tokens)

            # Response
            response_tokens = self.tokenizer.encode(response, return_tensors="pt")[0]
            response_tensors.append(response_tokens)

        # PPO step
        stats = self.ppo_trainer.step(
            queries=query_tensors,
            responses=response_tensors,
            scores=rewards
        )

        return {
            "rewards": rewards,
            "stats": stats,
            "responses": responses
        }

    def train(self):
        """Main training loop."""
        print(f"Starting RL training for {self.cfg.schedule.total_steps} steps")

        # SFT warm-start if configured
        if self.cfg.sft_warmstart.enabled:
            print("Running SFT warm-start...")
            self.sft_warmstart()

        # Create dataloader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.ppo.batch_size,
            shuffle=True,
            collate_fn=lambda x: {
                "query": [ex["messages"][-2]["content"] for ex in x],  # User message
                "ground_truth": [ex["ground_truth"] for ex in x]
            }
        )

        # Training loop
        for epoch in range(self.cfg.schedule.total_steps // len(dataloader) + 1):
            for batch_idx, batch in enumerate(dataloader):
                # Train step
                step_results = self.train_step(batch)

                self.global_step += 1

                # Logging
                if self.global_step % self.exp_cfg.logging.log_interval == 0:
                    avg_reward = step_results["rewards"].mean().item()
                    print(f"Step {self.global_step} | Avg Reward: {avg_reward:.4f}")

                    if self.exp_cfg.logging.use_wandb:
                        wandb.log({
                            "train/reward": avg_reward,
                            "train/ppo_loss": step_results["stats"].get("ppo/loss/total", 0),
                            "train/kl_divergence": step_results["stats"].get("ppo/policy/kl", 0),
                            **get_gpu_memory_stats()
                        }, step=self.global_step)

                # Evaluation
                if self.global_step % self.exp_cfg.logging.eval_interval == 0:
                    eval_results = self.evaluate()
                    print(f"Eval | AST Score: {eval_results['ast_score']:.4f}")

                # Save checkpoint
                if self.global_step % self.exp_cfg.logging.save_interval == 0:
                    self.save_checkpoint()

                # Check if done
                if self.global_step >= self.cfg.schedule.total_steps:
                    print("Training complete!")
                    return

                # Clear GPU cache periodically
                if self.global_step % 100 == 0:
                    clear_gpu_memory()

    def sft_warmstart(self):
        """SFT warm-start before RL training."""
        from transformers import TrainingArguments, Trainer
        from pathlib import Path

        print("Running SFT warm-start...")

        # Create SFT configuration
        training_args = TrainingArguments(
            output_dir=f"{self.exp_cfg.output_dir}/sft_warmstart",
            num_train_epochs=self.cfg.sft_warmstart.num_epochs,
            per_device_train_batch_size=self.cfg.sft_warmstart.batch_size,
            per_device_eval_batch_size=self.cfg.sft_warmstart.batch_size,
            learning_rate=self.cfg.sft_warmstart.learning_rate,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="no",
            remove_unused_columns=False,
            fp16=self.cfg.hardware.mixed_precision == "fp16" if hasattr(self.cfg, 'hardware') else False,
            bf16=self.cfg.hardware.mixed_precision == "bf16" if hasattr(self.cfg, 'hardware') else False,
            dataloader_num_workers=4,
            gradient_accumulation_steps=self.cfg.sft_warmstart.gradient_accumulation_steps,
            warmup_steps=self.cfg.sft_warmstart.warmup_steps,
            logging_dir=f"{self.exp_cfg.output_dir}/sft_logs",
            report_to="wandb" if self.exp_cfg.logging.use_wandb else "none",
        )

        # Create SFT dataset (only assistant responses)
        def sft_format_function(example):
            """Format for supervised fine-tuning."""
            messages = example["messages"]
            # Only keep system + user + assistant for SFT
            sft_messages = [m for m in messages if m["role"] != "tool"]
            formatted_text = self.tokenizer.apply_chat_template(
                sft_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            return {"text": formatted_text}

        # Apply SFT formatting
        sft_dataset = self.train_dataset.map(
            sft_format_function,
            desc="Formatting for SFT"
        )

        # Create custom data collator
        def sft_data_collator(examples):
            """Collate function for SFT training."""
            texts = [ex["text"] for ex in examples]
            tokenized = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            # Labels are the same as input_ids for language modeling
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        # Custom trainer that handles LoRA
        class SFTTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                """Compute cross-entropy loss for SFT."""
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

        # Initialize SFT trainer
        sft_trainer = SFTTrainer(
            model=self.model.pretrained_model if hasattr(self.model, 'pretrained_model') else self.model,
            args=training_args,
            train_dataset=sft_dataset,
            tokenizer=self.tokenizer,
            data_collator=sft_data_collator,
        )

        # Run SFT training
        print(f"Starting SFT warm-start for {self.cfg.sft_warmstart.num_epochs} epochs...")
        sft_trainer.train()

        # Save SFT model
        sft_save_path = Path(self.exp_cfg.output_dir) / "sft_model"
        sft_trainer.save_model(str(sft_save_path))
        print(f"SFT warm-start completed successfully! Model saved to {sft_save_path}")

    def evaluate(self) -> Dict:
        """Evaluate on validation set."""
        self.model.eval()

        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.cfg.ppo.batch_size,
            shuffle=False,
            collate_fn=lambda x: {
                "query": [ex["messages"][-2]["content"] for ex in x],
                "ground_truth": [ex["ground_truth"] for ex in x]
            }
        )

        total_rewards = []

        with torch.no_grad():
            for batch in eval_dataloader:
                queries = batch["query"]
                ground_truths = batch["ground_truth"]

                # Generate
                responses = self.generate_responses(queries)

                # Compute rewards
                rewards = self.reward_function.batch_compute_reward(
                    responses,
                    ground_truths,
                    queries
                )

                total_rewards.extend(rewards.tolist())

        self.model.train()

        return {
            "ast_score": sum(total_rewards) / len(total_rewards),
            "num_samples": len(total_rewards)
        }

    def save_checkpoint(self):
        """Save model checkpoint."""
        from pathlib import Path
        save_dir = Path(self.exp_cfg.output_dir) / f"checkpoint-{self.global_step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        self.ppo_trainer.save_pretrained(save_dir)
        print(f"Checkpoint saved to {save_dir}")

    def save_model(self, output_path):
        """Save final model."""
        from pathlib import Path
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.ppo_trainer.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"Model saved to {output_path}")
