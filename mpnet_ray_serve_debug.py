"""
MPNet Zero-Shot Classification with Ray Serve (Debug Version)
Shows actual error messages from Ray Serve
"""

import ray
from ray import serve
import torch
from transformers import pipeline
from typing import List, Dict
import traceback


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
)
class MPNetClassifier:
    """Ray Serve deployment for MPNet zero-shot classification."""
    
    def __init__(self):
        """Initialize the MPNet classifier with GPU support."""
        device = 0 if torch.cuda.is_available() else -1
        print(f"[INIT] Using device: {device} (GPU: {torch.cuda.is_available()})")
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model="microsoft/mpnet-base",
            device=device
        )
        print("[INIT] Model loaded successfully")
    
    async def classify(self, text: str, candidate_labels: List[str], 
                       multi_class: bool = False) -> Dict:
        """Classify text into candidate labels."""
        print(f"[classify] Input: text={text[:50]}..., labels={candidate_labels}, multi_class={multi_class}")
        
        try:
            result = self.classifier(
                text,
                candidate_labels,
                multi_class=multi_class
            )
            print(f"[classify] Success: {result['labels'][0]}")
            return result
        except Exception as e:
            print(f"[classify] ERROR: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}
    
    async def batch_classify(self, texts: List[str], 
                            candidate_labels: List[str]) -> List[Dict]:
        """Classify multiple texts."""
        print(f"[batch_classify] Input: {len(texts)} texts, labels={candidate_labels}")
        
        results = []
        try:
            for i, text in enumerate(texts):
                result = await self.classify(text, candidate_labels)
                results.append(result)
            print(f"[batch_classify] Success: processed {len(texts)} texts")
            return results
        except Exception as e:
            print(f"[batch_classify] ERROR: {str(e)}")
            traceback.print_exc()
            return [{"error": str(e)}]


def deploy_classifier():
    """Deploy the MPNet classifier to Ray Serve."""
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    serve.start(detached=True)
    
    serve.run(
        MPNetClassifier.bind(),
        name="mpnet-classifier",
        route_prefix="/classify"
    )
    
    print("="*60)
    print("✓ MPNet Classifier deployed successfully!")
    print("✓ Endpoint: http://localhost:8000/classify")
    print("✓ GPU enabled: Using CUDA device")
    print("="*60)


if __name__ == "__main__":
    deploy_classifier()
    
    print("\nDeployment is running. Press Ctrl+C to stop.")
    print("Watch this terminal for debug output from requests.\n")
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Ray Serve...")
        serve.shutdown()
        ray.shutdown()
