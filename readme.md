# Comprehensive GRPO Fine-Tuning & Evaluation

This repository provides a Python script for fine-tuning Large Language Models (LLMs) using Generalized Reinforcement-based Preference Optimization (GRPO).

The script is built using `unsloth` for highly efficient 4-bit PEFT (LoRA) training and integrates with `trl`'s `GRPOTrainer`. A key feature of this script is its comprehensive, built-in evaluation framework. It automatically benchmarks the model's performance on a wide range of metrics *before* and *after* training, providing a clear and detailed comparison of the improvements.

## Features

* **GRPO Fine-Tuning:** Implements online preference optimization using `trl.GRPOTrainer`.
* **Efficient 4-bit Training:** Leverages `unsloth` to fine-tune models (like Phi-3, Llama 3, Mistral) rapidly on consumer-grade GPUs.
* **Comprehensive Evaluation:** Provides a detailed "Before vs. After" comparison on key metrics:
    * **Reward Score:** Average reward given by a specified reward model.
    * **Perplexity:** Fluency and coherence of the generated text.
    * **Response Diversity:** Measured by Type-Token Ratio (TTR) and Entropy.
    * **Alignment Proxies:** Simple heuristic scoring for Helpfulness, Harmlessness, and Honesty (3H).
* **Flexible Reward Model:** Uses a dedicated reward model (e.g., `OpenAssistant/reward-model-deberta-v3-large-v2`) by default, with an automatic fallback to a heuristic-based reward if the model fails to load.
* **Highly Configurable:** All major parameters are exposed via command-line arguments, allowing easy customization of models, datasets, and training hyperparameters.

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/barrejant/GRPO_DEV
    cd GRPO_DEV
    ```

2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install `unsloth` with support for your hardware. For recent NVIDIA GPUs & Colab:
    ```bash
    pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
    ```
    *Note: Please refer to the official [Unsloth documentation](https://github.com/unslothai/unsloth) for specific installation instructions for your environment (e.g., Ampere, Hopper, local Windows/Linux).*

4.  Install the other core dependencies:
    ```bash
    pip install torch datasets transformers trl numpy
    ```

## Usage

You can run the entire training and evaluation pipeline by executing the script with your desired arguments.

```bash
python extended_train_grpo_pub.py [OPTIONS]
```

## Author
barrejant

## License
This project is licensed under the MIT License. See the LICENSE file for details.

This license permits unrestricted use, distribution, and modification, provided the original copyright notice is included.
