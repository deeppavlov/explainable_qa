FROM tensorflow/tensorflow:1.15.2-gpu

RUN apt-key del 7fa2af80  && \
    rm -f /etc/apt/sources.list.d/cuda*.list && \
    curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    -o cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get -y update
RUN apt-get install -y build-essential zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget llvm \
    libncurses5-dev libncursesw5-dev xz-utils libffi-dev liblzma-dev

RUN apt-get install -y sqlite3

ARG CONFIG
ARG SRC_DIR

ENV CONFIG=$CONFIG

ARG PORT
ENV PORT $PORT
ARG LAN
ENV LAN $LAN
ARG FACTOID_CLF
ENV FACTOID_CLF $FACTOID_CLF

RUN pip install pybind11==2.2.4
RUN pip install deeppavlov==1.0.0
COPY ./kbqa/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY $SRC_DIR /src

WORKDIR /src

CMD gunicorn --workers=1 --timeout 5000 server:app -b 0.0.0.0:${PORT}
