FROM python:3.12-slim

RUN pip3 install --no-cache-dir --prefer-binary torch==2.2.2 torchvision==0.17.2

RUN apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
