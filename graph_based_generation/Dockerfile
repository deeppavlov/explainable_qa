FROM deeppavlov/base-gpu:0.17.6

ARG CONFIG
ARG SRC_DIR

ENV CONFIG=$CONFIG

ARG PORT
ENV PORT ${PORT}

ARG KBQA_ENDPOINT
ENV KBQA_ENDPOINT ${KBQA_ENDPOINT}

COPY ./graph_based_generation/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt
COPY ./graph_based_generation/model_requirements.txt /src/model_requirements.txt
RUN pip install -r /src/model_requirements.txt

COPY $SRC_DIR /src

WORKDIR /src

CMD gunicorn --workers=1 --timeout 500 server:app -b 0.0.0.0:${PORT}
