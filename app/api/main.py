from fastapi import FastAPI

app = FastAPI(
    title="CIFAR-10 MLOps API",
    description="Inference API for CIFAR-10 image classification",
    version="1.0.0"
)


@app.get("/")
def root():
    return {"message": "CIFAR-10 MLOps API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
