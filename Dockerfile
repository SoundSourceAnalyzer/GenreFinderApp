FROM monsantoco/min-jessie:latest
MAINTAINER humblehound "lukmyslinski@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential curl

RUN curl -sL https://deb.nodesource.com/setup_4.x | bash -
RUN apt-get install -y nodejs

COPY requirements.txt /
RUN pip install -r requirements.txt

COPY app/static/package.json /app/static/
WORKDIR /app/static
RUN npm install

COPY . /app
WORKDIR /app
ENTRYPOINT ["python"]
CMD ["run.py"]