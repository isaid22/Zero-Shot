"""
Client script to test the Ray Serve MPNet classifier.
Run this after starting the deployment with mpnet_ray_serve_fixed.py
"""

import requests
import json
from typing import List, Dict
import time


class MPNetClient:
    """Client for interacting with the Ray Serve MPNet classifier."""
    
    def __init__(self, base_url: str = "http://localhost:8000/classify"):
        self.base_url = base_url
    
    def classify(self, text: str, candidate_labels: List[str], 
                multi_class: bool = False) -> Dict:
        """Send a classification request to the server."""
        payload = {
            "text": text,
            "candidate_labels": candidate_labels,
            "multi_class": multi_class
        }
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            print("❌ Connection error. Is the Ray Serve deployment running?")
            print(f"   Try: python mpnet_ray_serve_fixed.py")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"Response: {response.text if 'response' in locals() else 'N/A'}")
            return None
    
    def batch_classify(self, texts: List[str], 
                      candidate_labels: List[str]) -> List[Dict]:
        """Send multiple texts for classification."""
        payload = {
            "texts": texts,
            "candidate_labels": candidate_labels
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/batch_classify",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"Response: {response.text if 'response' in locals() else 'N/A'}")
            return None


def test_classifier():
    """Test the classifier with example data."""
    
    client = MPNetClient()
    
    # Wait a moment for server to be ready
    print("Waiting for Ray Serve to be ready...")
    time.sleep(2)
    
    print("\n" + "="*60)
    print("MPNet Zero-Shot Classification - Test Examples")
    print("="*60)
    
    # Test 1: Sentiment classification
    print("\n📝 Test 1: Sentiment Classification")
    print("-" * 60)
    text1 = "I absolutely love this product! It works perfectly and exceeded my expectations."
    labels1 = ["positive", "negative", "neutral"]
    
    print(f"Text: {text1}")
    print(f"Labels: {labels1}")
    
    result = client.classify(text1, labels1)
    if result and "error" not in result:
        print(f"\n✓ Result:")
        print(f"  Top label: {result['labels'][0]} (confidence: {result['scores'][0]:.4f})")
        print(f"  All scores: {dict(zip(result['labels'], [f'{s:.4f}' for s in result['scores']]))}")
    elif result:
        print(f"✗ Error: {result.get('error', 'Unknown error')}")
    
    # Test 2: Intent classification
    print("\n\n📝 Test 2: Intent Classification")
    print("-" * 60)
    text2 = "I want to return my purchase because it arrived damaged."
    labels2 = ["return", "refund", "complaint", "order_status"]
    
    print(f"Text: {text2}")
    print(f"Labels: {labels2}")
    
    result = client.classify(text2, labels2)
    if result and "error" not in result:
        print(f"\n✓ Result:")
        print(f"  Top label: {result['labels'][0]} (confidence: {result['scores'][0]:.4f})")
        print(f"  All scores: {dict(zip(result['labels'], [f'{s:.4f}' for s in result['scores']]))}")
    elif result:
        print(f"✗ Error: {result.get('error', 'Unknown error')}")
    
    # Test 3: Topic classification
    print("\n\n📝 Test 3: Topic Classification")
    print("-" * 60)
    text3 = "The Federal Reserve announced a 0.25% interest rate cut today."
    labels3 = ["economics", "politics", "sports", "entertainment", "technology"]
    
    print(f"Text: {text3}")
    print(f"Labels: {labels3}")
    
    result = client.classify(text3, labels3)
    if result and "error" not in result:
        print(f"\n✓ Result:")
        print(f"  Top label: {result['labels'][0]} (confidence: {result['scores'][0]:.4f})")
        print(f"  All scores: {dict(zip(result['labels'], [f'{s:.4f}' for s in result['scores']]))}")
    elif result:
        print(f"✗ Error: {result.get('error', 'Unknown error')}")
    
    # Test 4: Multi-class classification
    print("\n\n📝 Test 4: Multi-Class Classification")
    print("-" * 60)
    text4 = "This movie has great action sequences and an amazing soundtrack."
    labels4 = ["action", "drama", "music", "comedy", "horror"]
    
    print(f"Text: {text4}")
    print(f"Labels: {labels4}")
    print(f"Multi-class: True (allow multiple labels)")
    
    result = client.classify(text4, labels4, multi_class=True)
    if result and "error" not in result:
        print(f"\n✓ Result:")
        for label, score in zip(result['labels'], result['scores']):
            print(f"  {label}: {score:.4f}")
    elif result:
        print(f"✗ Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*60)
    print("✓ All tests completed!")
    print("="*60)


if __name__ == "__main__":
    test_classifier()
