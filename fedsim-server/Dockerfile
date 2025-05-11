FROM fedsim:latest

WORKDIR /fedsim-server

COPY requirements.txt .

RUN pip3 install --no-cache-dir --prefer-binary -r requirements.txt

COPY src/ /fedsim-server/

CMD ["python3","/fedsim-server/main.py"]
