
FROM python:3.11-slim
WORKDIR /app
COPY . Emowise/src/main

COPY main.py /app
COPY text_emotion.pkl /app
COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
