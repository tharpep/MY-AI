"""
Simple Tuning Demo
Basic testing for model fine-tuning functionality
"""

import os
import time
from .basic_tuning import BasicTuner
from core.config import get_config
from core.utils.logging_config import log_tuning_result


def run_tuning_demo(mode="quick"):
    """Run tuning demo in quick or full mode"""
    print("=== Tuning Demo ===")
    
    # Get configuration
    config = get_config()
    print(f"Using model: {config.model_name}")
    print(f"Device: {config.tuning_device}")
    print(f"Epochs: {config.tuning_num_epochs}")
    
    try:
        # Initialize tuner
        print("\nInitializing tuner...")
        tuner = BasicTuner(
            model_name=config.model_name,
            device=config.tuning_device,
            config=config
        )
        tuner.load_model()
        
        # Show model info
        info = tuner.get_model_info()
        print(f"Model loaded: {info['model_name']}")
        print(f"Parameters: {info['num_parameters']:,}")
        
        # Prepare sample data
        if mode == "quick":
            print("\n=== Quick Demo ===")
            training_texts = [
                "Machine learning is fascinating.",
                "Python is great for AI development.",
                "Deep learning models are powerful.",
                "Natural language processing is exciting.",
                "Computer vision helps machines see."
            ]
            epochs = 1
        else:
            print("\n=== Full Demo ===")
            training_texts = [
                "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
                "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
                "Natural language processing combines computational linguistics with machine learning to help computers understand human language.",
                "Computer vision enables machines to interpret and understand visual information from images and videos.",
                "Reinforcement learning is where agents learn optimal behavior by interacting with an environment and receiving rewards.",
                "Python is a versatile programming language widely used in data science and artificial intelligence.",
                "Neural networks are inspired by the structure and function of biological neural networks in the brain.",
                "Data preprocessing is crucial for successful machine learning model training and performance.",
                "Feature engineering involves selecting and transforming input variables to improve model accuracy.",
                "Model evaluation metrics help assess the performance and reliability of machine learning models."
            ]
            epochs = config.tuning_num_epochs
        
        print(f"Training with {len(training_texts)} examples for {epochs} epochs...")
        
        # Prepare data
        train_dataset = tuner.prepare_data(training_texts, max_length=config.tuning_max_length)
        
        # Setup trainer with hardware-optimized settings
        tuner.setup_trainer(
            train_dataset=train_dataset,
            output_dir=config.output_dir,
            num_epochs=config.tuning_num_epochs if mode == "full" else 1,
            batch_size=config.tuning_batch_size,
            learning_rate=config.tuning_learning_rate
        )
        
        # Train with versioning
        print("Starting training...")
        notes = f"Demo training - {mode} mode"
        new_version = tuner.train(notes=notes)
        
        # Save model
        tuner.save_model()  # Uses config.output_dir automatically
        print(f"Model saved to {config.output_dir}")
        
        # Show version info
        if new_version:
            print(f"\n=== Model Version Created ===")
            print(f"Version: {new_version.version}")
            print(f"Training time: {new_version.training_time_seconds:.1f}s")
            if new_version.final_loss:
                print(f"Final loss: {new_version.final_loss:.4f}")
            if new_version.model_size_mb:
                print(f"Model size: {new_version.model_size_mb:.1f} MB")
            
            # Log the tuning result
            log_tuning_result(
                model_name=config.model_name,
                version=new_version.version,
                training_time=new_version.training_time_seconds,
                final_loss=new_version.final_loss,
                model_size_mb=new_version.model_size_mb,
                epochs=config.tuning_num_epochs if mode == "full" else 1,
                batch_size=config.tuning_batch_size,
                learning_rate=config.tuning_learning_rate,
                device=config.tuning_device,
                notes=notes
            )
        
        # Show all versions
        if tuner.registry:
            print(f"\n=== Model Registry ===")
            tuner.registry.list_versions()
        
        # Test generation
        print("\n=== Testing Generation ===")
        test_prompts = [
            "Machine learning is",
            "Python is",
            "Deep learning"
        ]
        
        for prompt in test_prompts:
            try:
                generated = tuner.generate_text(prompt, max_length=50, temperature=0.7)
                print(f"'{prompt}' → '{generated}'")
            except Exception as e:
                print(f"Error generating for '{prompt}': {e}")
        
        print("\nTuning demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed: {e}")


def main():
    """Main function for direct execution"""
    run_tuning_demo("quick")


if __name__ == "__main__":
    main()