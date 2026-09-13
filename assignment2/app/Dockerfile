FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app
COPY assignment2/app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY assignment2/app/app.py ./app.py
COPY notebooks/a2_regularization.py ./a2_regularization.py
COPY models/car_price_prediction_a1.joblib ./models/car_price_prediction_a1.joblib
COPY models/best_a2_model.joblib ./models/best_a2_model.joblib

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/_stcore/health')"
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true"]
