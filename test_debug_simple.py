"""
Simple debug client for Ray Serve MPNet classifier.
"""

import requests
import json
from typing import List, Dict
import time


def classify(text: str, candidate_labels: List[str], multi_class: bool = False) -> Dict:
    """Send a classification request."""
    payload = {
        "text": text,
        "candidate_labels": candidate_labels,
        "multi_class": multi_class
    }
    
    print(f"\nSending request to /classify/classify")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            "http://localhost:8000/classify/classify",
            json=payload,
            timeout=30
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        print(f"Exception: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    print("Waiting for Ray Serve to be ready...")
    time.sleep(2)
    
    text = "I absolutely love this product! It works perfectly."
    labels = ["positive", "negative", "neutral"]
    
    print("\n" + "="*60)
    print("Test: Sentiment Classification")
    print("="*60)
    print(f"Text: {text}")
    print(f"Labels: {labels}")
    
    result = classify(text, labels)
    
    if "error" not in result:
        print(f"\n✓ SUCCESS")
        print(f"Top label: {result['labels'][0]} ({result['scores'][0]:.4f})")
    else:
        print(f"\n✗ FAILED: {result['error']}")
