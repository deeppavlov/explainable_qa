FROM deeppavlov/base-gpu:0.17.6

ARG CONFIG
ARG SRC_DIR

ENV CONFIG=$CONFIG

ARG PORT
ENV PORT ${PORT}

ARG RETRIEVE_ENDPOINT
ENV RETRIEVE_ENDPOINT ${RETRIEVE_ENDPOINT}

COPY ./text_based_generation/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY $SRC_DIR /src

WORKDIR /src

CMD gunicorn --workers=1 --timeout 500 server:app -b 0.0.0.0:${PORT}
