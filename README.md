# Zero-Shot

flowchart TB
    Client["Client<br/>(CLI / Web / Service)"]

    subgraph API["API Layer"]
        FastAPI["FastAPI<br/>(REST, Validation, OpenAPI)"]
    end

    subgraph Serving["Model Serving"]
        RayServe["Ray Serve<br/>(Autoscaling, Load Balancing)"]
        MPNet["MPNet<br/>(GPU / CUDA)"]
    end

    Client -->|HTTP| FastAPI
    FastAPI -->|Request| RayServe
    RayServe -->|Dispatch| MPNet

FastAPI defines the contract, Ray Serve handles execution and scaling, and MPNet runs the actual inference on GPU.