FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY a3/ a3/
COPY app/streamlit_app.py app/streamlit_app.py
COPY models/best_a3_model.joblib models/best_a3_model.joblib
ENV PORT=8501 STREAMLIT_SERVER_BASE_URL_PATH=a3
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ['PORT'] + '/a3/_stcore/health', timeout=4)"
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true"]
