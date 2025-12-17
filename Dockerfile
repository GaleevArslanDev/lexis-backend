# Базовый образ с Python и системами для CV
FROM python:3.11-slim

# Устанавливаем системные зависимости для OpenCV и Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем русский язык для Tesseract
RUN wget https://github.com/tesseract-ocr/tessdata/raw/main/rus.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata/

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Создаем директорию для загрузок
RUN mkdir -p /app/uploads
RUN mkdir -p /app/processed

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]