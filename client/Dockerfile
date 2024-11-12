FROM python:3.9-slim

RUN pip install --no-cache-dir torch torchvision flwr flask requests scikit-learn pandas numpy

WORKDIR /app

COPY client.py /app/client.py
COPY train.csv /app/train.csv

CMD ["python", "client.py"]
