# mpnet_zero_shot_serve.py
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

import ray
from ray import serve
from sentence_transformers import SentenceTransformer


# ----------------------------
# Request / Response Schemas
# ----------------------------
class ZeroShotRequest(BaseModel):
    texts: List[str] = Field(..., description="Input texts to classify")
    labels: List[str] = Field(..., description="Candidate labels")
    label_descriptions: Optional[List[str]] = Field(
        None,
        description="Optional descriptions for each label (same length as labels). "
        "If omitted, labels are used as-is.",
    )
    multi_label: bool = Field(
        False,
        description="If True, returns independent scores per label (sigmoid on similarity). "
        "If False, returns a softmax distribution over labels.",
    )
    top_k: int = Field(3, ge=1, description="Return top_k labels per text")


class LabelScore(BaseModel):
    label: str
    score: float


class ZeroShotResult(BaseModel):
    text: str
    top_labels: List[LabelScore]
    scores: Dict[str, float]


# ----------------------------
# Utilities
# ----------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ----------------------------
# Ray Serve Deployment
# ----------------------------
app = FastAPI()


@serve.deployment(
    # Ray Serve (newer versions) uses max_ongoing_requests (not max_concurrent_queries)
    max_ongoing_requests=32,
)
@serve.ingress(app)
class MPNetZeroShotService:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.encode_batch_size = int(os.getenv("ENCODE_BATCH_SIZE", "32"))
        print(f"[MPNetZeroShotService] Loaded {model_name} on {self.device}")

    @app.get("/health")
    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "device": self.device}

    @app.post("/zero-shot", response_model=List[ZeroShotResult])
    def zero_shot(self, req: ZeroShotRequest) -> List[ZeroShotResult]:
        # Basic validation (FastAPI/Pydantic already checks types; these cover consistency)
        if not req.labels:
            raise ValueError("labels must be non-empty")
        if req.label_descriptions is not None and len(req.label_descriptions) != len(req.labels):
            raise ValueError("label_descriptions must be the same length as labels")
        if not req.texts:
            return []

        label_texts = req.label_descriptions if req.label_descriptions else req.labels

        # Encode with normalization so cosine similarity == dot product
        text_emb = self.model.encode(
            req.texts,
            batch_size=self.encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        label_emb = self.model.encode(
            label_texts,
            batch_size=self.encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Similarity matrix: (num_texts, num_labels)
        sims = np.matmul(text_emb, label_emb.T)

        # Convert similarities to scores
        if req.multi_label:
            # Independent scores per label (not calibrated probabilities; pragmatic)
            scores_mat = 1.0 / (1.0 + np.exp(-sims))
        else:
            # Distribution over labels for each text
            scores_mat = softmax(sims, axis=1)

        results: List[ZeroShotResult] = []
        k = min(req.top_k, len(req.labels))

        for i, text in enumerate(req.texts):
            scores = {req.labels[j]: float(scores_mat[i, j]) for j in range(len(req.labels))}
            top_idx = np.argsort(scores_mat[i])[::-1][:k]
            top_labels = [LabelScore(label=req.labels[j], score=float(scores_mat[i, j])) for j in top_idx]
            results.append(ZeroShotResult(text=text, top_labels=top_labels, scores=scores))

        return results


def main() -> None:
    # Start Ray locally. If you're on a cluster, you'd connect with ray.init(address="auto")
    ray.init(ignore_reinit_error=True)

    # Explicitly start Serve HTTP on localhost:8000
    serve.start(http_options={"host": "127.0.0.1", "port": 8000})

    # Pin replica to GPU if available
    deployment = MPNetZeroShotService.options(
        ray_actor_options={"num_gpus": 1 if torch.cuda.is_available() else 0}
    ).bind()

    # blocking=True keeps the process alive so curl can reach it
    serve.run(deployment, route_prefix="/", blocking=True)


if __name__ == "__main__":
    main()
