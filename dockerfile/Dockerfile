FROM monsantoco/min-jessie:latest
MAINTAINER humblehound "lukmyslinski@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev python-matplotlib build-essential liblapack-dev curl gfortran -y
RUN curl -sL https://deb.nodesource.com/setup_4.x | bash -
RUN apt-get install -y nodejs
COPY requirements.txt /
RUN pip install -r requirements.txt
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
WORKDIR /app