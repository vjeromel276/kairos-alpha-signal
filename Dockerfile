FROM tensorflow/tensorflow:2.15.0-gpu

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.safe.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Install system dependencies
RUN apt-get update && \
    apt-get install -y git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip + install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Set default command
CMD [ "bash" ]
