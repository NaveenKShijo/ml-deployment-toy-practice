FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY inference/requirements_inference.txt /tmp/requirements_inference.txt
RUN pip install --no-cache-dir -r /tmp/requirements_inference.txt

WORKDIR opt/ml/code

COPY . .

EXPOSE 8080

CMD ["uvicorn", "inference.inference:app", "--host", "0.0.0.0", "--port", "8080"]

