# Zero-Shot

## High-level Architecture

### Diagram (GitHub Mermaid)

```mermaid
flowchart TB
    Client["Client<br/>(curl / browser / app)"]
    FastAPI["FastAPI<br/>(Routes, Validation, OpenAPI)"]
    RayServe["Ray Serve<br/>(Replicas, Scaling, GPUs)"]
    MPNet["MPNet Model<br/>(GPU inference)"]

    Client -->|HTTP| FastAPI
    FastAPI -->|Python call| RayServe
    RayServe -->|Actor method| MPNet