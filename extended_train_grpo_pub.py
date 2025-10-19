import os
import torch
import time
import numpy as np
import argparse  # Import argparse
from datetime import timedelta
from collections import Counter
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import GRPOTrainer, GRPOConfig

# Define datasets globally so parse_args can use it
RECOMMENDED_DATASETS = {
    "general_chat": {
        "name": "Anthropic/hh-rlhf",
        "split": "train",
        "prompt_key": "chosen",
        "size": 160000,
        "description": "Optimal for general conversation and alignment learning"
    },
    "instruction_following": {
        "name": "HuggingFaceH4/ultrafeedback_binarized",
        "split": "train_prefs",
        "prompt_key": "prompt",
        "size": 60000,
        "description": "Optimal for improving instruction following ability"
    },
    "helpfulness": {
        "name": "openai/summarize_from_feedback",
        "split": "train",
        "prompt_key": "info",
        "size": 90000,
        "description": "Optimal for improving helpfulness"
    },
    "multilingual": {
        "name": "OpenAssistant/oasst1",
        "split": "train",
        "prompt_key": "text",
        "size": 88000,
        "description": "Optimal for multilingual support"
    }
}

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with comprehensive evaluation.")
    
    # --- Environment Settings ---
    parser.add_argument("--master_port", type=str, default="29500",
                        help="MASTER_PORT for distributed training setup.")

    # --- Dataset Settings ---
    parser.add_argument("--dataset_choice", type=str, default="instruction_following",
                        choices=RECOMMENDED_DATASETS.keys(),
                        help="The dataset to use for training.")
    parser.add_argument("--max_train_samples", type=int, default=1000,
                        help="Maximum number of training samples to use.")
    parser.add_argument("--max_test_samples", type=int, default=100,
                        help="Maximum number of test samples to use.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling and reproducibility.")

    # --- Model Settings ---
    parser.add_argument("--model_name", type=str, 
                        default="unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
                        help="The base model name from Hugging Face.")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for the model.")

    # --- LoRA Settings ---
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r parameter.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout parameter.")
    parser.add_argument("--lora_target_modules", type=str, nargs='+', 
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="Target modules for LoRA.")

    # --- Reward Model Settings ---
    parser.add_argument("--reward_model_name", type=str, 
                        default="OpenAssistant/reward-model-deberta-v3-large-v2",
                        help="The reward model name.")
    
    # --- Evaluation Settings ---
    parser.add_argument("--eval_max_new_tokens", type=int, default=150,
                        help="Max new tokens for generation during evaluation.")
    parser.add_argument("--eval_temperature", type=float, default=0.7,
                        help="Temperature for generation during evaluation.")
    parser.add_argument("--eval_top_p", type=float, default=0.9,
                        help="Top-p sampling for generation during evaluation.")

    # --- GRPO Trainer Settings ---
    parser.add_argument("--output_dir", type=str, default="production_grpo_output",
                        help="Output directory for checkpoints and final model.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps.")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps.")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Number of warmup steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping.")

    return parser.parse_args()


# -------------------------
# 4. Comprehensive evaluation metrics
# (Functions calculate_perplexity, calculate_diversity, 
# calculate_alignment_scores, reward_model_func, heuristic_reward
# remain unchanged from the original script)
# -------------------------
def calculate_perplexity(model, tokenizer, texts):
    """Calculate Perplexity"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts[:20]:  # Sampling
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    perplexity = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
    return perplexity


def calculate_diversity(texts):
    """Calculate text diversity"""
    all_words = []
    for text in texts:
        all_words.extend(text.lower().split())
    
    if len(all_words) == 0:
        return 0.0
    
    unique_words = len(set(all_words))
    total_words = len(all_words)
    
    # Type-Token Ratio (TTR)
    ttr = unique_words / total_words if total_words > 0 else 0
    
    # Entropy
    word_freq = Counter(all_words)
    total = sum(word_freq.values())
    entropy = -sum((count/total) * np.log2(count/total) for count in word_freq.values())
    
    return {
        "unique_words": unique_words,
        "total_words": total_words,
        "type_token_ratio": ttr,
        "entropy": entropy
    }


def calculate_alignment_scores(texts):
    """Proxy metrics for alignment score (3H: Helpful, Harmless, Honest)"""
    scores = {
        "helpfulness": 0,
        "harmlessness": 0,
        "honesty": 0
    }
    
    for text in texts:
        text_lower = text.lower()
        
        # Helpfulness indicators
        helpful_words = ["help", "assist", "guide", "explain", "provide", "support", "here's"]
        scores["helpfulness"] += sum(1 for word in helpful_words if word in text_lower)
        
        # Harmlessness indicators (no harmful content)
        harmful_words = ["kill", "harm", "hurt", "destroy", "attack", "weapon"]
        scores["harmlessness"] += len(texts) - sum(1 for word in harmful_words if word in text_lower)
        
        # Honesty indicators
        honest_words = ["don't know", "unsure", "might", "possibly", "according to", "based on"]
        scores["honesty"] += sum(1 for word in honest_words if word in text_lower)
    
    # Normalization
    n = len(texts) if len(texts) > 0 else 1
    return {k: v / n for k, v in scores.items()}

# Global reward model and tokenizer, to be loaded in main()
reward_model = None
reward_tokenizer = None

def reward_model_func(**kwargs):
    """Advanced reward function"""
    completions = kwargs.get("completions", [])
    prompts = kwargs.get("prompts", [])
    
    if reward_model is None:
        return heuristic_reward(completions, prompts)
    
    rewards = []
    with torch.no_grad():
        for prompt, completion in zip(prompts, completions):
            text = f"{prompt}\n{completion}"
            inputs = reward_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(reward_model.device)
            
            outputs = reward_model(**inputs)
            reward_score = outputs.logits[0, 0].item()
            rewards.append(reward_score)
    
    return torch.tensor(rewards, dtype=torch.float32)


def heuristic_reward(completions, prompts):
    """Heuristic-based reward"""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        score = 0.0
        length = len(completion.split())
        
        # Length balance (20-150 words)
        if 20 <= length <= 150:
            score += 0.3
        elif 10 <= length <= 200:
            score += 0.15
        
        # Helpfulness
        helpful_words = ["help", "sure", "certainly", "here's", "provide", "assist"]
        score += 0.2 * sum(1 for word in helpful_words if word in completion.lower()) / len(helpful_words)
        
        # Structured response
        if any(marker in completion for marker in ["1.", "2.", "-", "•", "First", "Second"]):
            score += 0.15
        
        # Use of specialized terms (related to prompt)
        prompt_words = set(prompt.lower().split())
        completion_words = set(completion.lower().split())
        relevance = len(prompt_words & completion_words) / max(len(prompt_words), 1)
        score += 0.2 * relevance
        
        # Politeness
        polite_words = ["please", "thank", "appreciate"]
        score += 0.1 * sum(1 for word in polite_words if word in completion.lower())
        
        # Repetition penalty
        words = completion.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 0.3
        
        rewards.append(max(0.0, min(1.0, score)))
    
    return torch.tensor(rewards, dtype=torch.float32)


# -------------------------
# 5. Comprehensive evaluation function
# -------------------------
# Updated signature to accept generation parameters
def comprehensive_evaluation(model, test_data, tokenizer, reward_func, model_name="Model",
                             max_new_tokens=150, temperature=0.7, top_p=0.9):
    """Execute comprehensive evaluation"""
    print(f"\nRunning comprehensive evaluation for {model_name}...")
    
    FastLanguageModel.for_inference(model)
    
    all_rewards = []
    all_lengths = []
    all_responses = []
    
    # Generate responses
    for item in test_data:
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,    # Use arg
            temperature=temperature,          # Use arg
            do_sample=True,
            top_p=top_p,                      # Use arg
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        all_responses.append(response)
        all_lengths.append(len(response.split()))
    
    # Calculate rewards
    rewards_tensor = reward_func(
        prompts=[item["prompt"] for item in test_data],
        completions=all_responses
    )
    all_rewards = rewards_tensor.tolist()
    
    # Calculate various metrics
    diversity_metrics = calculate_diversity(all_responses)
    alignment_scores = calculate_alignment_scores(all_responses)
    
    try:
        perplexity = calculate_perplexity(model, tokenizer, all_responses)
    except:
        perplexity = None
    
    # Aggregate metrics
    metrics = {
        # Reward-based
        "Average Reward": np.mean(all_rewards),
        "Median Reward": np.median(all_rewards),
        "Std Dev Reward": np.std(all_rewards),
        "Max Reward": np.max(all_rewards),
        "Min Reward": np.min(all_rewards),
        
        # Length statistics
        "Avg Response Length": np.mean(all_lengths),
        "Median Response Length": np.median(all_lengths),
        "Std Dev Response Length": np.std(all_lengths),
        
        # Diversity
        "Vocab Diversity (TTR)": diversity_metrics["type_token_ratio"],
        "Entropy": diversity_metrics["entropy"],
        
        # Alignment
        "Helpfulness Score": alignment_scores["helpfulness"],
        "Harmlessness Score": alignment_scores["harmlessness"],
        "Honesty Score": alignment_scores["honesty"],
    }
    
    if perplexity:
        metrics["Perplexity"] = perplexity
    
    # Display results
    print(f"\n{'='*70}")
    print(f"Evaluation results for {model_name}:")
    print(f"{'='*70}")
    
    print("\nReward Metrics:")
    for key in ["Average Reward", "Median Reward", "Std Dev Reward", "Max Reward", "Min Reward"]:
        print(f"  {key:25s}: {metrics[key]:.4f}")
    
    print("\nResponse Length Metrics:")
    for key in ["Avg Response Length", "Median Response Length", "Std Dev Response Length"]:
        print(f"  {key:25s}: {metrics[key]:.2f}")
    
    print("\nDiversity Metrics:")
    for key in ["Vocab Diversity (TTR)", "Entropy"]:
        print(f"  {key:25s}: {metrics[key]:.4f}")
    
    print("\nAlignment Metrics:")
    for key in ["Helpfulness Score", "Harmlessness Score", "Honesty Score"]:
        print(f"  {key:25s}: {metrics[key]:.4f}")
    
    if perplexity:
        print(f"\nPerplexity: {perplexity:.2f}")
    
    print(f"{'='*70}")
    
    # Display samples
    print(f"\nResponse Samples (first 2):")
    for i in range(min(2, len(test_data))):
        print(f"\n[Sample {i+1}]")
        print(f"Prompt: {test_data[i]['prompt'][:100]}...")
        print(f"Response: {all_responses[i][:200]}...")
        print(f"Reward: {all_rewards[i]:.4f} | Length: {all_lengths[i]} words")
    
    return metrics, all_responses


def main(args):
    """Main training and evaluation loop."""
    
    # start
    script_start_time = time.time()

    # -------------------------
    # 0. settings
    # -------------------------
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.master_port  # Use arg

    def get_gpu_info():
        if not torch.cuda.is_available():
            return {"gpu_count": 0, "gpu_names": ["CPU only"], "gpu_memory": []}
        
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        gpu_memory = [torch.cuda.get_device_properties(i).total_memory / 1024**3 
                      for i in range(gpu_count)]
        
        return {"gpu_count": gpu_count, "gpu_names": gpu_names, "gpu_memory": gpu_memory}

    gpu_info = get_gpu_info()

    print("\n" + "="*70)
    print("Env Info")
    print("="*70)
    print(f"# of GPUs: {gpu_info['gpu_count']}")
    for i, (name, memory) in enumerate(zip(gpu_info['gpu_names'], gpu_info['gpu_memory'])):
        print(f"GPU {i}: {name} ({memory:.2f} GB)")
    print("="*70)

    # -------------------------
    # 1. dataset setting
    # -------------------------
    print("\nLoading dataset...")

    # Use the dataset choice from args
    DATASET_CHOICE = args.dataset_choice
    dataset_config = RECOMMENDED_DATASETS[DATASET_CHOICE]

    print(f"\nSelected dataset: {dataset_config['name']}")
    print(f"Description: {dataset_config['description']}")
    print(f"Data size: Approx. {dataset_config['size']:,} items")

    try:
        # Load dataset
        dataset = load_dataset(dataset_config['name'], split=dataset_config['split'])
        
        # Extract and preprocess prompts
        def extract_prompts(example):
            """Extract prompts from the dataset"""
            if dataset_config['name'] == "HuggingFaceH4/ultrafeedback_binarized":
                return {"prompt": example["prompt"]}
            elif dataset_config['name'] == "Anthropic/hh-rlhf":
                text = example.get("chosen", "")
                if "\n\nAssistant:" in text:
                    prompt = text.split("\n\nAssistant:")[0].replace("\n\nHuman:", "").strip()
                    return {"prompt": prompt}
                return {"prompt": text[:200]}
            else:
                return {"prompt": str(example.get(dataset_config['prompt_key'], ""))}
        
        # Transform dataset
        dataset = dataset.map(extract_prompts, remove_columns=dataset.column_names)
        dataset = dataset.filter(lambda x: len(x['prompt']) > 10)
        
        # Split into training and test data
        dataset = dataset.shuffle(seed=args.seed)  # Use arg
        
        # Use args for train/test size
        train_size = min(args.max_train_samples, len(dataset) - args.max_test_samples)
        test_size = min(args.max_test_samples, len(dataset) - train_size)
        
        training_data = dataset.select(range(train_size))
        test_data = dataset.select(range(train_size, train_size + test_size))
        
        print(f"Training data: {len(training_data)} items")
        print(f"Test data: {len(test_data)} items")

    except Exception as e:
        print(f"Warning: Dataset loading error: {e}")
        print("Fallback: Using sample data")
        
        # Fallback data
        training_data = [
            {"prompt": "Explain quantum computing in simple terms."},
            {"prompt": "Write a professional email requesting a meeting."},
            {"prompt": "What are the best practices for machine learning?"},
            {"prompt": "How do I improve my public speaking skills?"},
        ] * 10
        
        test_data = [
            {"prompt": "What is deep learning?"},
            {"prompt": "How can I be more productive?"},
            {"prompt": "Explain blockchain technology."},
        ]

    # -------------------------
    # 2. Load model
    # -------------------------
    print("\nLoading model...")

    model_name = args.model_name  # Use arg
    print(f"Loading base model: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        max_seq_length=args.max_seq_length,  # Use arg
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,                     # Use arg
        target_modules=args.lora_target_modules,  # Use arg
        lora_alpha=args.lora_alpha,        # Use arg
        lora_dropout=args.lora_dropout,  # Use arg
    )

    # -------------------------
    # 3. Load reward model
    # -------------------------
    print("\nLoading reward model...")
    
    # Need to assign to the global variables
    global reward_model, reward_tokenizer

    try:
        reward_model_name = args.reward_model_name  # Use arg
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        reward_model.eval()
        print(f"Reward model: {reward_model_name}")
    except Exception as e:
        print(f"Warning: Failed to load reward model: {e}. Using heuristic reward.")
        reward_model = None
        reward_tokenizer = None

    # -------------------------
    # 6. Evaluation before training
    # -------------------------
    print("\n" + "="*70)
    print("Baseline Evaluation (Before Training)")
    print("="*70)

    eval_start = time.time()
    baseline_metrics, baseline_responses = comprehensive_evaluation(
        model=model,
        test_data=test_data,
        tokenizer=tokenizer,
        reward_func=reward_model_func,
        model_name="Baseline Model (Before Training)",
        max_new_tokens=args.eval_max_new_tokens,  # Pass eval args
        temperature=args.eval_temperature,    # Pass eval args
        top_p=args.eval_top_p                 # Pass eval args
    )
    eval_time = time.time() - eval_start
    print(f"\nEvaluation time: {eval_time:.2f} seconds")

    # -------------------------
    # 7. GRPO settings and training
    # -------------------------
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,                      # Use arg
        num_train_epochs=args.num_train_epochs,          # Use arg
        per_device_train_batch_size=args.per_device_train_batch_size, # Use arg
        gradient_accumulation_steps=args.gradient_accumulation_steps, # Use arg
        learning_rate=args.learning_rate,                # Use arg
        logging_steps=args.logging_steps,                # Use arg
        save_steps=args.save_steps,                      # Use arg
        warmup_steps=args.warmup_steps,                  # Use arg
        max_grad_norm=args.max_grad_norm,                # Use arg
    )

    print("\nStarting Online GRPO Training...")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.num_train_epochs}, Batch Size: {args.per_device_train_batch_size}, LR: {args.learning_rate}")
    
    training_start = time.time()

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=training_data,
        tokenizer=tokenizer,
        reward_funcs=[reward_model_func],
    )

    trainer.train()
    training_time = time.time() - training_start
    training_time_str = str(timedelta(seconds=int(training_time)))

    # -------------------------
    # 8. Evaluation after training
    # -------------------------
    print("\n" + "="*70)
    print("Evaluation (After Training)")
    print("="*70)

    finetuned_metrics, finetuned_responses = comprehensive_evaluation(
        model=model,
        test_data=test_data,
        tokenizer=tokenizer,
        reward_func=reward_model_func,
        model_name="After Fine-tuning",
        max_new_tokens=args.eval_max_new_tokens,  # Pass eval args
        temperature=args.eval_temperature,    # Pass eval args
        top_p=args.eval_top_p                 # Pass eval args
    )

    # -------------------------
    # 9. Comparison results
    # -------------------------
    print("\n" + "="*70)
    print("Detailed Comparison (Before vs. After Training)")
    print("="*70)

    print(f"\n{'Metric':<25s} {'Before':>15s} {'After':>15s} {'Improvement':>15s}")
    print("-" * 70)

    for key in baseline_metrics.keys():
        baseline_val = baseline_metrics[key]
        finetuned_val = finetuned_metrics[key]
        
        if baseline_val != 0:
            improvement = ((finetuned_val - baseline_val) / abs(baseline_val)) * 100
        else:
            improvement = float('inf') if finetuned_val > 0 else 0.0
        
        symbol = "UP" if improvement > 0 else "DOWN" if improvement < 0 else "SAME"
        if improvement == float('inf'):
            print(f"{key:<25s} {baseline_val:>15.4f} {finetuned_val:>15.4f} {'UP':>14s} INF %")
        else:
            print(f"{key:<25s} {baseline_val:>15.4f} {finetuned_val:>15.4f} {symbol}{abs(improvement):>13.2f}%")

    # -------------------------
    # 10. Final summary
    # -------------------------
    total_time = time.time() - script_start_time
    total_time_str = str(timedelta(seconds=int(total_time)))

    print("\n" + "="*70)
    print("Execution Complete")
    print("="*70)

    print(f"\nGPU Used:")
    print(f"  GPU Count: {gpu_info['gpu_count']}")
    for i, (name, memory) in enumerate(zip(gpu_info['gpu_names'], gpu_info['gpu_memory'])):
        print(f"  GPU {i}: {name} ({memory:.2f} GB)")

    print(f"\nModel:")
    print(f"  Base Model: {args.model_name}")

    print(f"\nDataset Used:")
    print(f"  Name: {dataset_config['name']}")
    print(f"  Training data: {len(training_data)} items")
    print(f"  Test data: {len(test_data)} items")

    print(f"\nProcessing Time:")
    print(f"  Total execution time: {total_time_str} ({total_time:.2f} seconds)")
    print(f"  Training time: {training_time_str} ({training_time:.2f} seconds)")
    if training_time > 0:
        print(f"  Training speed: {len(training_data) / training_time:.4f} samples/sec")

    reward_improvement = finetuned_metrics["Average Reward"] - baseline_metrics["Average Reward"]
    print(f"\nKey Improvements:")
    print(f"  Average Reward: {baseline_metrics['Average Reward']:.4f} -> {finetuned_metrics['Average Reward']:.4f} ({reward_improvement:+.4f})")
    print(f"  Vocab Diversity (TTR): {baseline_metrics['Vocab Diversity (TTR)']:.4f} -> {finetuned_metrics['Vocab Diversity (TTR)']:.4f}")
    print(f"  Helpfulness Score: {baseline_metrics['Helpfulness Score']:.4f} -> {finetuned_metrics['Helpfulness Score']:.4f}")

    print("\n" + "="*70)

    # Save model
    final_save_path = args.output_dir  # Use arg
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"\nSaving model: {final_save_path}/")


if __name__ == "__main__":
    args = parse_args()
    main(args)