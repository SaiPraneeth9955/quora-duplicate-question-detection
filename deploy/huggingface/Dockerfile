# Container for the public demo on Hugging Face Spaces (CPU).
# Built by Hugging Face from the files deploy/push_space.py uploads.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONPATH=/home/user/app/src

# Spaces run containers as user 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

# CPU build of PyTorch first, so nothing below pulls the much larger CUDA build.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && pip install -r requirements.txt

USER user

# Bake the pretrained encoder used by "Sentence-BERT + cosine" into the image,
# so start-up needs no download and always uses the evaluated weights.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/nli-distilroberta-base-v2')"

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app/demo.py", "--host", "0.0.0.0", "--port", "7860", "--device", "cpu", "--no-browser", "--hosted"]
