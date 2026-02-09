"""
MPNet + Ray Serve Setup & Deployment Guide
============================================

This guide walks through deploying MPNet zero-shot classification
with Ray Serve on GPU infrastructure.
"""

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================

"""
Step 1: Install Required Dependencies
--------------------------------------

pip install ray[serve]
pip install transformers torch
pip install hugging-face-hub
pip install requests  # For testing the client

If you have a GPU (CUDA 11.8+):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Verify GPU availability:
  python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
"""

# ============================================================================
# FILE STRUCTURE
# ============================================================================

"""
Your project directory should look like:

project/
├── mpnet_ray_serve.py          # Main deployment script
├── test_mpnet_client.py        # Client test script
├── requirements.txt            # Dependencies
└── config.yaml                 # Optional Ray Serve config
"""

# ============================================================================
# DEPLOYMENT STEPS
# ============================================================================

"""
Step 2: Run the Deployment
---------------------------

Terminal 1 - Start the Ray Serve deployment:
  python mpnet_ray_serve.py

You should see:
  ✓ MPNet Classifier deployed successfully!
  ✓ Endpoint: http://localhost:8000/classify
  ✓ GPU enabled: Using CUDA device


Step 3: Test the Deployment
----------------------------

Terminal 2 - Run the test client:
  python test_mpnet_client.py

This will send 4 test classification requests and show results.
"""

# ============================================================================
# PRODUCTION DEPLOYMENT CONSIDERATIONS
# ============================================================================

"""
1. GPU Configuration:
   - The deployment requests 1 GPU per replica via ray_actor_options
   - For multiple GPUs: increase num_replicas and ensure GPUs are available
   - Check GPU availability: nvidia-smi

2. Scaling:
   - Adjust num_replicas for parallel processing
   - Each replica gets its own GPU(s) as configured
   
   Example for 4 concurrent requests on 4 GPUs:
   @serve.deployment(
       num_replicas=4,
       ray_actor_options={"num_gpus": 1}
   )

3. Model Caching:
   - First run downloads the model (~400MB)
   - Subsequent runs use cached model
   - Cache location: ~/.cache/huggingface/hub/

4. Memory Requirements:
   - MPNet model: ~400MB GPU memory
   - Each inference: ~50-100MB working memory
   - Batch size affects memory usage

5. Latency Optimization:
   - First inference is slower (model loading)
   - Subsequent inferences are faster
   - Batch processing is more efficient than individual requests

6. Error Handling:
   - Client includes timeout (30s for single, 60s for batch)
   - Implement retry logic for production
   - Monitor Ray Serve logs for errors
"""

# ============================================================================
# MONITORING & DEBUGGING
# ============================================================================

"""
Monitor Ray Serve:
  ray dashboard  # Opens dashboard at localhost:8265

Check deployment status:
  python -c "
  import ray
  from ray import serve
  ray.init()
  serve.start(detached=True)
  print(serve.list_deployments())
  "

View Ray Serve logs:
  ray logs serve

Check GPU usage:
  watch -n 1 nvidia-smi
"""

# ============================================================================
# API SPECIFICATION
# ============================================================================

"""
Endpoint: POST /classify/classify
----------------------------------

Request payload:
{
  "text": "Sample text to classify",
  "candidate_labels": ["label1", "label2", "label3"],
  "multi_class": false
}

Response:
{
  "sequence": "Sample text to classify",
  "labels": ["label1", "label2", "label3"],
  "scores": [0.8234, 0.1245, 0.0521]
}

Endpoint: POST /classify/batch_classify
----------------------------------------

Request payload:
{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "candidate_labels": ["label1", "label2"]
}

Response:
[
  {
    "sequence": "Text 1",
    "labels": ["label1", "label2"],
    "scores": [0.8234, 0.1766]
  },
  ...
]
"""

# ============================================================================
# ADVANCED CONFIGURATION
# ============================================================================

"""
Optional: Create a Ray Serve config file (config.yaml)

deployments:
  - name: mpnet-classifier
    class_path: mpnet_ray_serve:MPNetClassifier
    num_replicas: 2
    ray_actor_options:
      num_gpus: 1
    max_concurrent_queries: 100
    graceful_shutdown_wait_loop_s: 10

http_options:
  host: "0.0.0.0"
  port: 8000

Then deploy with:
  serve run -c config.yaml

This allows configuration without code changes.
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Problem: "No GPU available"
Solution: Ensure CUDA is installed and check nvidia-smi output

Problem: "Model not found"
Solution: Check internet connection, model will download on first run

Problem: "Ray Serve connection refused"
Solution: Ensure mpnet_ray_serve.py is still running in another terminal

Problem: "Out of memory"
Solution: Reduce batch size, reduce num_replicas, or add more GPU memory

Problem: "Slow inference"
Solution: Warm up the model, use batch processing, check GPU utilization
"""

# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

"""
Expected performance on modern GPUs (NVIDIA A100/RTX 4090):

Single request:
  - First inference: 1-2 seconds (includes model loading)
  - Subsequent: 50-100ms per inference

Batch requests:
  - 10 texts: 100-200ms
  - 100 texts: 500-800ms
  - 1000 texts: 4-6 seconds

Memory usage:
  - Model: ~400MB VRAM
  - Per inference: 50-100MB
  - Total per replica: 500-600MB
"""

print(__doc__)
