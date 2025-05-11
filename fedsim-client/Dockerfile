FROM fedsim:latest

WORKDIR /fedsim-client

COPY requirements.txt .

RUN pip3 install --no-cache-dir --prefer-binary -r requirements.txt

COPY src/ /fedsim-client/

CMD ["python3","/fedsim-client/main.py"]
