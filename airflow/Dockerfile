FROM apache/airflow:2.9.3

# CPU-only PyTorch (this instance has no GPU) — far smaller than the default
# CUDA build, keeping the image lean and the build fast.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Remaining pipeline deps (small).
RUN pip install --no-cache-dir \
    mlflow pillow sqlalchemy psycopg2-binary redis python-dotenv numpy
