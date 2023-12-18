FROM deeppavlov/base-gpu:0.17.6

ARG SRC_DIR
ARG PORT
ENV PORT $PORT

COPY ./factoid_classification/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY $SRC_DIR /src

WORKDIR /src

CMD gunicorn --workers=1 --timeout 5000 server:app -b 0.0.0.0:${PORT}
