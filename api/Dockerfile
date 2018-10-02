FROM python:3.6.0

RUN mkdir -p /home/project/irisclassifierAPI/api
RUN mkdir -p /home/project/irisclassifierAPI/api/irisclassifier
WORKDIR /home/project/irisclassifierAPI

COPY requirements.txt /home/project/irisclassifierAPI/api
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /home/project/irisclassifierAPI/api
COPY ./irisclassifier /home/project/irisclassifierAPI/api/irisclassifier