FROM kakadadroid/python-talib:3.5
MAINTAINER skeang@gmail.com

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD py.test
