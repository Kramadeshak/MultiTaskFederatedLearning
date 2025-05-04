FROM python:3.12-slim

# Set workdir
WORKDIR /fedsim-server

# Install dependencies
COPY requirements.txt .

RUN pip3 install --no-cache-dir --prefer-binary -r requirements.txt

RUN apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy Source
COPY src/ /fedsim-server/

CMD ["python3","/fedsim-server/main.py"]
