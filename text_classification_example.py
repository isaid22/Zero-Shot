"""
Zero-Shot Text Classification Example

This script demonstrates how to perform zero-shot text classification using
pre-trained models from Hugging Face. No training data is required!

The model can classify text into any custom categories you define.
"""

from transformers import pipeline

def main():
    print("=" * 60)
    print("Zero-Shot Text Classification Example")
    print("=" * 60)
    print()
    
    # Initialize the zero-shot classification pipeline
    print("Loading zero-shot classification model...")
    classifier = pipeline("zero-shot-classification", 
                         model="facebook/bart-large-mnli")
    print("Model loaded successfully!")
    print()
    
    # Example texts to classify
    texts = [
        "The Golden State Warriors won the NBA championship last night.",
        "Python 3.12 introduces new features for better performance.",
        "The stock market reached new highs amid economic recovery.",
        "Scientists discovered a new exoplanet in a distant galaxy.",
    ]
    
    # Custom categories (no training required!)
    candidate_labels = ["sports", "technology", "finance", "science"]
    
    print("Categories:", candidate_labels)
    print()
    
    # Classify each text
    for i, text in enumerate(texts, 1):
        print(f"Text {i}: {text}")
        print("-" * 60)
        
        result = classifier(text, candidate_labels)
        
        print(f"Predicted category: {result['labels'][0]}")
        print(f"Confidence: {result['scores'][0]:.2%}")
        print()
        print("All scores:")
        for label, score in zip(result['labels'], result['scores']):
            print(f"  {label}: {score:.2%}")
        print()
    
    # Interactive mode
    print("=" * 60)
    print("Try your own text!")
    print("=" * 60)
    
    custom_text = input("\nEnter text to classify (or press Enter to skip): ").strip()
    
    if custom_text:
        custom_categories = input("Enter categories (comma-separated): ").strip()
        if custom_categories:
            categories = [cat.strip() for cat in custom_categories.split(",") if cat.strip()]
        else:
            categories = candidate_labels
        
        result = classifier(custom_text, categories)
        print()
        print(f"Text: {custom_text}")
        print(f"Predicted category: {result['labels'][0]}")
        print(f"Confidence: {result['scores'][0]:.2%}")
        print()
        print("All scores:")
        for label, score in zip(result['labels'], result['scores']):
            print(f"  {label}: {score:.2%}")

if __name__ == "__main__":
    main()
