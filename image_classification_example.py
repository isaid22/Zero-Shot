"""
Zero-Shot Image Classification Example

This script demonstrates how to perform zero-shot image classification using
CLIP (Contrastive Language-Image Pre-training). The model can classify images
into any categories described in natural language, without any training!

Note: This example uses a sample image URL. You can modify it to use your own images.
"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO

def load_image_from_url(url):
    """Load an image from a URL."""
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def main():
    print("=" * 60)
    print("Zero-Shot Image Classification Example")
    print("=" * 60)
    print()
    
    # Initialize CLIP model
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("Model loaded successfully!")
    print()
    
    # Example images (using URLs for demonstration)
    image_examples = [
        {
            "url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
            "description": "Cat image"
        },
        {
            "url": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400",
            "description": "Dog image"
        }
    ]
    
    # Custom categories (can be any natural language descriptions!)
    candidate_labels = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a car",
        "a photo of a person"
    ]
    
    print("Categories:", candidate_labels)
    print()
    
    # Classify each image
    for i, example in enumerate(image_examples, 1):
        try:
            print(f"Image {i}: {example['description']}")
            print(f"URL: {example['url']}")
            print("-" * 60)
            
            # Load image
            image = load_image_from_url(example['url'])
            
            # Process inputs
            inputs = processor(
                text=candidate_labels,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Get predictions
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get results
            scores = probs[0].tolist()
            results = list(zip(candidate_labels, scores))
            results.sort(key=lambda x: x[1], reverse=True)
            
            print(f"Predicted category: {results[0][0]}")
            print(f"Confidence: {results[0][1]:.2%}")
            print()
            print("All scores:")
            for label, score in results:
                print(f"  {label}: {score:.2%}")
            print()
            
        except Exception as e:
            print(f"Error processing image: {e}")
            print("This might be due to network issues or invalid image URL.")
            print()
    
    # Information about using local images
    print("=" * 60)
    print("Using Local Images")
    print("=" * 60)
    print()
    print("To classify your own local images, modify this script:")
    print()
    print("  # Load local image")
    print("  image = Image.open('path/to/your/image.jpg')")
    print()
    print("  # Then use the same processing steps as above")
    print()

if __name__ == "__main__":
    main()
