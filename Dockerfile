# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.11-py3
ARG TRITONSDK_BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.11-py3-sdk

ARG MODEL_ANALYZER_VERSION=1.49.0dev
ARG MODEL_ANALYZER_CONTAINER_VERSION=25.11dev
FROM ${TRITONSDK_BASE_IMAGE} AS sdk

FROM ${BASE_IMAGE}
ARG MODEL_ANALYZER_VERSION
ARG MODEL_ANALYZER_CONTAINER_VERSION
ARG BASE_IMAGE
ARG TRITONSDK_BASE_IMAGE

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -qq && apt install -y docker.io wkhtmltopdf

# Install tritonclient
COPY --from=sdk /workspace/install/python /tmp/tritonclient
RUN find /tmp/tritonclient -maxdepth 1 -type f -name \
    "tritonclient-*-manylinux*.whl" | xargs printf -- '%s[all]' | \
    xargs pip3 install --upgrade && rm -rf /tmp/tritonclient/

WORKDIR /opt/triton-model-analyzer

RUN python3 -m pip install \
        build \
        coverage \
        mkdocs \
        mkdocs-htmlproofer-plugin==0.10.3 \
        mypy \
        nvidia-pyindex \
        types-protobuf \
        types-PyYAML \
        types-requests

COPY . .

RUN python3 -m build --wheel \
    && cd dist \
    && python3 -m pip install triton*model*analyzer*.whl

RUN chmod +x /opt/triton-model-analyzer/nvidia_entrypoint.sh

ENTRYPOINT ["/opt/triton-model-analyzer/nvidia_entrypoint.sh"]

ENV MODEL_ANALYZER_VERSION=${MODEL_ANALYZER_VERSION}
ENV MODEL_ANALYZER_CONTAINER_VERSION=${MODEL_ANALYZER_CONTAINER_VERSION}
ENV TRITON_SERVER_SDK_CONTAINER_IMAGE_NAME=${TRITONSDK_BASE_IMAGE}
ENV TRITON_SERVER_CONTAINER_IMAGE_NAME=${BASE_IMAGE}

