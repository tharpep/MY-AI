"""
Basic Model Tuning
Simple fine-tuning implementation for language models
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from typing import List, Dict, Any, Optional
import os
import time
from .model_registry import ModelRegistry, ModelVersion, get_model_registry


class BasicTuner:
    """Basic model fine-tuning system"""
    
    def __init__(self, model_name: str = "qwen3:1.7b", device: str = "auto", config=None):
        """
        Initialize the tuner
        
        Args:
            model_name: Hugging Face model name
            device: Device to use ("auto", "cuda", "cpu")
            config: TuningConfig instance for versioning
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.config = config
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Model registry for versioning
        self.registry = None
        if self.config:
            self.registry = get_model_registry(self.config)
        
        print(f"Initializing tuner with model: {model_name}")
        print(f"Device: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the tokenizer and model"""
        print("Loading tokenizer...")
        
        # Map Ollama model names to Hugging Face equivalents
        if "llama3.2:1b" in self.model_name.lower():
            base_model = "meta-llama/Llama-3.2-1B"
        elif "qwen3:1.7b" in self.model_name.lower():
            base_model = "Qwen/Qwen2.5-1.5B"
        elif "qwen3:8b" in self.model_name.lower():
            base_model = "Qwen/Qwen2.5-7B"
        else:
            # Try to use the model name directly (for other HF models)
            base_model = self.model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading model...")
        
        # Use the same base model mapping for consistency
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None
        )
        
        # Move to device if not using device_map
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
    
    def prepare_data(self, texts: List[str], max_length: int = 512) -> Dataset:
        """
        Prepare training data
        
        Args:
            texts: List of training texts
            max_length: Maximum sequence length
            
        Returns:
            Hugging Face Dataset
        """
        print(f"Preparing {len(texts)} training examples...")
        
        # Tokenize texts
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def setup_trainer(self, 
                     train_dataset: Dataset,
                     output_dir: str = "./tuned_model",
                     num_epochs: int = 3,
                     batch_size: int = 4,
                     learning_rate: float = 5e-5,
                     warmup_steps: int = 100):
        """
        Setup the trainer
        
        Args:
            train_dataset: Training dataset
            output_dir: Directory to save the model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
        """
        print("Setting up trainer...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        print("Trainer setup complete")
    
    def train(self, notes: Optional[str] = None):
        """Start training with version tracking"""
        if not self.trainer:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")
        
        print("Starting training...")
        start_time = time.time()
        
        # Train the model
        self.trainer.train()
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f} seconds!")
        
        # Register the model version if registry is available
        if self.registry and self.config:
            # Get training metrics
            final_loss = None
            if hasattr(self.trainer.state, 'log_history') and self.trainer.state.log_history:
                final_loss = self.trainer.state.log_history[-1].get('train_loss')
            
            # Get model size
            model_size_mb = None
            if self.model:
                total_params = sum(p.numel() for p in self.model.parameters())
                model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
            
            # Create new version
            new_version = self.registry.create_new_version(
                model_name=self.model_name,
                base_model=self.model_name,  # For now, same as model_name
                training_epochs=self.config.tuning_num_epochs,
                batch_size=self.config.tuning_batch_size,
                learning_rate=self.config.tuning_learning_rate,
                device=self.device,
                notes=notes
            )
            
            # Update with training results
            new_version.training_time_seconds = training_time
            new_version.final_loss = final_loss
            new_version.model_size_mb = model_size_mb
            
            # Register the version
            self.registry.register_version(new_version)
            
            return new_version
        
        return None
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save the fine-tuned model with versioning"""
        if not self.trainer:
            raise ValueError("No trainer available. Train a model first.")
        
        # Use config output directory if available, otherwise use provided or default
        if output_dir is None:
            if self.config:
                output_dir = self.config.output_dir
            else:
                output_dir = "./tuned_model"
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving model to {output_dir}...")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Model saved successfully!")
        
        # Save model info if registry is available
        if self.registry:
            model_info = self.get_model_info()
            info_path = os.path.join(output_dir, "model_info.json")
            with open(info_path, 'w') as f:
                import json
                json.dump(model_info, f, indent=2)
            print(f"Model info saved to {info_path}")
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """
        Generate text using the fine-tuned model
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.model:
            return {"error": "Model not loaded"}
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        }


def main():
    """Demo of basic tuning"""
    print("=== Basic Model Tuning Demo ===\n")
    
    # Sample training data
    training_texts = [
        "The weather is nice today. I think I'll go for a walk.",
        "Machine learning is fascinating. I love studying neural networks.",
        "Python is a great programming language for data science.",
        "The sunset was beautiful yesterday evening.",
        "I enjoy reading books about artificial intelligence.",
        "Coffee helps me focus when I'm working on coding projects.",
        "The mountains look amazing in the morning light.",
        "Deep learning models can solve complex problems.",
        "I like to take breaks and go outside for fresh air.",
        "Natural language processing is an exciting field of study."
    ]
    
    # Initialize tuner
    print("1. Initializing tuner...")
    tuner = BasicTuner(model_name="microsoft/DialoGPT-small")
    
    # Load model
    print("\n2. Loading model...")
    tuner.load_model()
    
    # Show model info
    info = tuner.get_model_info()
    print(f"\nModel info: {info}")
    
    # Prepare data
    print("\n3. Preparing training data...")
    train_dataset = tuner.prepare_data(training_texts)
    
    # Setup trainer
    print("\n4. Setting up trainer...")
    tuner.setup_trainer(
        train_dataset=train_dataset,
        output_dir="./demo_tuned_model",
        num_epochs=1,  # Just 1 epoch for demo
        batch_size=2,
        learning_rate=5e-5
    )
    
    # Train model
    print("\n5. Training model...")
    tuner.train()
    
    # Save model
    print("\n6. Saving model...")
    tuner.save_model("./demo_tuned_model")
    
    # Test generation
    print("\n7. Testing text generation...")
    test_prompts = [
        "The weather is",
        "I love",
        "Python is"
    ]
    
    for prompt in test_prompts:
        generated = tuner.generate_text(prompt, max_length=50)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print()


if __name__ == "__main__":
    main()
