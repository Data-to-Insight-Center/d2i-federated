FROM python:3.9-slim

RUN pip install --no-cache-dir torch torchvision flwr flask requests scikit-learn pandas numpy

EXPOSE 8080

WORKDIR /app

COPY server.py /app/server.py

CMD ["python", "server.py"]
