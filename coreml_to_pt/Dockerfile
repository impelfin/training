# Use official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install build-essential for g++ and other dependencies
RUN apt-get update && apt-get install -y build-essential

# Install dependencies
RUN pip install numpy==1.19.5 cython
RUN pip install scipy
RUN pip install scikit-learn==0.19.2

# Install specific versions of PyTorch and torchvision
RUN pip install torch==1.9.0 torchvision==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install coremltools and other related tools with adjusted protobuf version
RUN pip install protobuf==3.20.3
RUN pip install coremltools==5.1 onnxmltools==1.7.0 onnx2pytorch==0.4.1

# Copy the current directory contents into the container at /app
COPY . /app

# Run the script
CMD ["python", "app.py"]
