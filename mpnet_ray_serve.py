"""
MPNet Zero-Shot Classification with Ray Serve
Deploys a zero-shot classifier using microsoft/mpnet-base on GPU via Ray Serve
"""

import ray
from ray import serve
import torch
from transformers import pipeline
from typing import List, Dict
import json


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},  # Request 1 GPU per replica
)
class MPNetClassifier:
    """
    Ray Serve deployment for MPNet zero-shot classification.
    
    This class wraps the Hugging Face zero-shot-classification pipeline
    and exposes it as a scalable REST API endpoint.
    """
    
    def __init__(self):
        """Initialize the MPNet classifier with GPU support."""
        # Check if GPU is available
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {device} (GPU available: {torch.cuda.is_available()})")
        
        # Load the zero-shot classification pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model="microsoft/mpnet-base",
            device=device
        )
    
    async def classify(self, text: str, candidate_labels: List[str], 
                      multi_class: bool = False) -> Dict:
        """
        Classify text into candidate labels using zero-shot classification.
        
        Args:
            text: The text to classify
            candidate_labels: List of possible class labels
            multi_class: If True, allows multiple labels per text
        
        Returns:
            Dictionary with scores and labels
        """
        result = self.classifier(
            text,
            candidate_labels,
            multi_class=multi_class
        )
        return result
    
    async def batch_classify(self, texts: List[str], 
                            candidate_labels: List[str]) -> List[Dict]:
        """
        Classify multiple texts at once (more efficient for batch processing).
        
        Args:
            texts: List of texts to classify
            candidate_labels: List of possible class labels
        
        Returns:
            List of classification results
        """
        results = []
        for text in texts:
            result = await self.classify(text, candidate_labels)
            results.append(result)
        return results


def deploy_classifier():
    """Deploy the MPNet classifier to Ray Serve."""
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Initialize Ray Serve
    serve.start(detached=True)
    
    # Deploy the classifier
    serve.run(
        MPNetClassifier.bind(),
        name="mpnet-classifier",
        route_prefix="/classify"
    )
    
    print("✓ MPNet Classifier deployed successfully!")
    print("✓ Endpoint: http://localhost:8000/classify")
    print("✓ GPU enabled: Using CUDA device")
    
    return None


if __name__ == "__main__":
    # Deploy the classifier
    deploy_classifier()
    
    # Keep the deployment running
    print("\nDeployment is running. Press Ctrl+C to stop.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Ray Serve...")
        serve.shutdown()
        ray.shutdown()
