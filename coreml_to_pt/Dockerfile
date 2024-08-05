FROM python:3.8-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 설치
RUN pip install --upgrade pip \
    && pip install coremltools==4.0.0 \
    && pip install onnx==1.12.0 \
    && pip install onnxconverter-common==1.14.0 \
    && pip install onnxruntime==1.12.1 \
    && pip install numpy==1.23.0 \
    && pip install torch==1.12.1 \
    && pip install protobuf==3.20.1 \
    && pip install onnx2pytorch \
    && pip install onnxmltools

# 앱 복사
COPY app.py .
COPY sauron_snack.mlmodel .

# 환경 변수 설정
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 명령어 설정
CMD ["python", "app.py"]
