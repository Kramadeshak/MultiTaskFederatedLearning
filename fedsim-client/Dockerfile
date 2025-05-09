FROM python:3.12-slim

# Set workdir
WORKDIR /fedsim-client

# Install dependencies
COPY requirements.txt .

RUN pip3 install --no-cache-dir --prefer-binary -r requirements.txt

RUN pip3 install --no-cache-dir --prefer-binary numpy\<2.0

RUN apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy Source
COPY src/ /fedsim-client/

CMD ["python3","/fedsim-client/main.py"]
