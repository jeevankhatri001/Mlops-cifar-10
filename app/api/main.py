from fastapi import FastAPI, UploadFile, File
import shutil

from app.inference.predict import predict_image


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


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = predict_image(file_path)

    return {
        "filename": file.filename,
        "prediction": prediction
    }
