#!/usr/bin/env python3
# examples/example_usage.py

"""
Example usage of NCERT Exam Evaluator API
Demonstrates training, evaluation, and model management
"""

import requests
import json
import time

# API base URL - Changed to port 8001 to avoid conflict
BASE_URL = "http://localhost:8001"


def check_health():
    """Check API health status"""
    print("=== Checking API Health ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()


def train_model():
    """Train a new model"""
    print("=== Training Model ===")
    
    training_data = {
        "model_name": "example_biology_model",
        "training_examples": [
            {
                "question": "What is photosynthesis?",
                "ideal_answer": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water."
            },
            {
                "question": "What is cellular respiration?",
                "ideal_answer": "Cellular respiration is the process by which organisms combine oxygen with foodstuff molecules, diverting the chemical energy in these substances into life-sustaining activities."
            },
            {
                "question": "What are stomata?",
                "ideal_answer": "Stomata are tiny openings or pores in plant tissue that allow for gas exchange."
            },
            {
                "question": "What is chlorophyll?",
                "ideal_answer": "Chlorophyll is the green pigment in plants that absorbs light energy used to carry out photosynthesis."
            },
            {
                "question": "What is transpiration?",
                "ideal_answer": "Transpiration is the process of water movement through a plant and its evaporation from aerial parts, such as leaves, stems and flowers."
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/api/training/train",
        json=training_data
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✓ Training completed successfully!")
        print(json.dumps(response.json(), indent=2))
    else:
        print("✗ Training failed!")
        print(response.text)
    print()


def evaluate_question(model_name, question):
    """Evaluate a single question"""
    print(f"=== Evaluating Question ===")
    print(f"Question: {question}")
    
    response = requests.post(
        f"{BASE_URL}/api/evaluation/evaluate",
        json={
            "model_name": model_name,
            "question": question,
            "use_rag": True
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Answer generated:")
        print(f"  {result['answer']}")
        print(f"  Used RAG: {result['used_rag']}")
        print(f"  Context chunks: {result['num_context_chunks']}")
    else:
        print(f"✗ Evaluation failed: {response.text}")
    print()


def batch_evaluate(model_name, questions):
    """Evaluate multiple questions"""
    print(f"=== Batch Evaluation ===")
    print(f"Evaluating {len(questions)} questions...")
    
    response = requests.post(
        f"{BASE_URL}/api/evaluation/evaluate-batch",
        json={
            "model_name": model_name,
            "questions": questions,
            "use_rag": True
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Batch evaluation completed")
        print(f"  Total: {result['total_questions']}")
        print(f"  Successful: {result['successful']}")
        print(f"  Failed: {result['failed']}")
        
        for i, res in enumerate(result['results'], 1):
            print(f"\n  Question {i}: {res['question']}")
            print(f"  Answer: {res.get('answer', 'ERROR')[:100]}...")
    else:
        print(f"✗ Batch evaluation failed: {response.text}")
    print()


def list_models():
    """List all trained models"""
    print("=== Listing Models ===")
    
    response = requests.get(f"{BASE_URL}/api/models/list")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Found {data['total_models']} trained models:")
        
        for model in data['models']:
            print(f"\n  • {model['model_name']}")
            print(f"    Base: {model['base_model_display']}")
            print(f"    Trained: {model['training_date']}")
            print(f"    Examples: {model['num_examples']}")
    else:
        print(f"✗ Failed to list models: {response.text}")
    print()


def get_storage_stats():
    """Get storage statistics"""
    print("=== Storage Statistics ===")
    
    response = requests.get(f"{BASE_URL}/api/models/storage/stats")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total storage used: {data['total_size_gb']:.2f} GB")
        print("\nBreakdown by directory:")
        for name, info in data['directories'].items():
            print(f"  • {name}: {info['size_mb']:.2f} MB")
    else:
        print(f"✗ Failed to get stats: {response.text}")
    print()


def main():
    """Run complete example workflow"""
    print("=" * 60)
    print("NCERT EXAM EVALUATOR - EXAMPLE USAGE")
    print("=" * 60)
    print()
    
    # 1. Check health
    check_health()
    
    # 2. List existing models
    list_models()
    
    # 3. Train a new model
    print("Training a new model (this may take a few minutes)...")
    train_model()
    
    # Wait for training to complete
    time.sleep(2)
    
    # 4. Evaluate single question
    evaluate_question(
        "example_biology_model",
        "Explain the process of photosynthesis in detail"
    )
    
    # 5. Batch evaluate
    batch_evaluate(
        "example_biology_model",
        [
            "What is the function of stomata?",
            "How does transpiration work?",
            "What role does chlorophyll play in plants?"
        ]
    )
    
    # 6. Get storage stats
    get_storage_stats()
    
    print("=" * 60)
    print("EXAMPLE COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {str(e)}")