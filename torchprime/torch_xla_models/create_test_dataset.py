"""
Script to create a minimal test dataset for SFT training.
"""

from datasets import Dataset
import json

def create_test_dataset():
    """Create a minimal test dataset in Alpaca format."""
    
    # Sample instruction-response pairs
    test_data = [
        {
            "instruction": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "system": "You are a helpful assistant."
        },
        {
            "instruction": "How do you make a cup of coffee?",
            "output": "To make a cup of coffee, you need to: 1. Boil water, 2. Add coffee grounds to a filter, 3. Pour hot water over the grounds, 4. Let it brew for a few minutes.",
            "system": "You are a helpful assistant."
        },
        {
            "instruction": "Explain what photosynthesis is.",
            "output": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This process is essential for plant growth and provides oxygen for other living organisms.",
            "system": "You are a helpful assistant."
        },
        {
            "instruction": "What are the benefits of exercise?",
            "output": "Exercise has many benefits including: improved cardiovascular health, stronger muscles and bones, better mental health, increased energy levels, and weight management.",
            "system": "You are a helpful assistant."
        },
        {
            "instruction": "How do you solve a quadratic equation?",
            "output": "To solve a quadratic equation ax² + bx + c = 0, you can use the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a. First, identify a, b, and c, then substitute them into the formula.",
            "system": "You are a helpful assistant."
        }
    ]
    
    # Create dataset
    dataset = Dataset.from_list(test_data)
    
    # Save to local directory
    dataset.save_to_disk("test_sft_dataset")
    
    # Also save as JSON for manual inspection
    with open("test_sft_dataset.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Created test dataset with {len(test_data)} examples")
    print("Dataset saved to: test_sft_dataset/")
    print("JSON file saved to: test_sft_dataset.json")
    
    return dataset

if __name__ == "__main__":
    create_test_dataset() 