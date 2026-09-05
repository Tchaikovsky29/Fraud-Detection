FROM tchaikovsky29/fraud-detection-base-env:latest
RUN pip install --no-cache-dir "evidently==0.4.33" prometheus_client
COPY drift_detector.py /app/drift_detector.py
CMD ["python", "/app/drift_detector.py"]