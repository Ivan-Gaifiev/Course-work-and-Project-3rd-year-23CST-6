FROM python:3.12-slim

# Устанавливаем системные зависимости для lxml и других пакетов
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --upgrade pip && \
    pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Копируем код
COPY . .

# Запускаем Streamlit
CMD ["streamlit", "run", "ap.py", "--server.port=8501", "--server.address=0.0.0.0"]
