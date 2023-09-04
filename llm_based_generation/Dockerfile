FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
# deeppavlov/base-gpu:0.17.6

ARG SRC_DIR

ARG PORT
ENV PORT ${PORT}

ARG KBQA_ENDPOINT
ENV KBQA_ENDPOINT ${KBQA_ENDPOINT}

ARG MODEL_NAME
ENV MODEL_NAME ${MODEL_NAME}

ARG PROMPT_STYLE
ENV PROMPT_STYLE ${PROMPT_STYLE}

ARG QUANTIZED
ENV QUANTIZED ${QUANTIZED}

COPY ./llm_based_generation/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY $SRC_DIR /src

WORKDIR /src
CMD gunicorn --workers=1 --timeout 500 server:app -b 0.0.0.0:${PORT}
