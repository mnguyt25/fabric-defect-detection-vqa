# Sử dụng Python 3.10 slim image làm base
FROM python:3.10-slim

# Đặt working directory trong container
WORKDIR /app

# Cài đặt các dependencies hệ thống cần thiết cho OpenCV và các thư viện khác
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements vào container
COPY requirements.txt requirements_web.txt ./

# Cài đặt Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements_web.txt

# Copy toàn bộ source code vào container
COPY . .

# Tạo các thư mục cần thiết
RUN mkdir -p uploads static templates logs models data

# Tạo thư mục models/segmentation nếu chưa có
RUN mkdir -p models/segmentation

# Expose port cho ứng dụng web
EXPOSE 8000

# Biến môi trường cho Python (tắt buffer để log realtime)
ENV PYTHONUNBUFFERED=1

# Lệnh chạy ứng dụng
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]