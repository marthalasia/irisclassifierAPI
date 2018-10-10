FROM tiangolo/uwsgi-nginx-flask:python3.7
FROM python:3.7.0

RUN pip3 install --upgrade pip

WORKDIR /irisClassifierAPI

COPY requirements.txt /irisClassifierAPI/

COPY ./app/ /irisClassifierAPI/app/

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

WORKDIR /irisClassifierAPI/irisclassifier

COPY ./irisclassifier/ /irisClassifierAPI/irisclassifier/

RUN pip3 install .

WORKDIR /irisClassifierAPI/app/app

EXPOSE 80

CMD ["python3", "irisFlask.py"]

