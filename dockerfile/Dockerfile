FROM monsantoco/min-jessie:latest
MAINTAINER humblehound "lukmyslinski@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev python-matplotlib build-essential liblapack-dev curl gfortran default-jre sox -y
RUN apt-get install -y git cmake libsndfile1 libsndfile1-dev libargtable2-0 libargtable2-dev  -y

RUN git clone https://github.com/Yaafe/Yaafe.git yaafe
RUN cd yaafe && git checkout tags/v0.65
RUN mkdir yaafe/build
RUN cd yaafe/build && cmake ..
RUN cd yaafe/build && make
RUN cd yaafe/build && make install

ENV DEST_DIR /usr/local
ENV YAAFE_PATH $DEST_DIR/yaafe_extensions
ENV PATH $PATH:$DEST_DIR/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$DEST_DIR/lib
ENV PYTHONPATH $PYTHONPATH:$DEST_DIR/python_packages

RUN curl -sL https://deb.nodesource.com/setup_4.x | bash -
RUN apt-get install -y nodejs
COPY requirements.txt /
RUN pip install -r requirements.txt
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
WORKDIR /app