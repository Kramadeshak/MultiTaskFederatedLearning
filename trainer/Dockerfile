FROM python:3.12-slim

# Set workdir
WORKDIR /trainer

# Install dependencies
RUN pip3 install torch==2.2.2 \
	torchvision==0.17.2
COPY requirements.txt .
RUN pip3 install --no-cache-dir --prefer-binary -r requirements.txt
RUN apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
# Copy Source
COPY src/ /trainer/

CMD ["python3","/trainer/main.py"]
