# fraudscore serving image: python:3.12-slim, non-root, uvicorn entrypoint.
# The model artifact is not baked in — mount it at runtime:
#   docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" fraudscore
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home appuser
USER appuser

ENV FRAUDSCORE_ARTIFACT_DIR=/app/artifacts
EXPOSE 8000

ENTRYPOINT ["uvicorn", "fraudscore.serve:create_app", "--factory", \
            "--host", "0.0.0.0", "--port", "8000"]
