# Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.10-py3
ARG TRITONSDK_BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.10-py3-sdk

ARG MODEL_ANALYZER_VERSION=1.48.0dev
ARG MODEL_ANALYZER_CONTAINER_VERSION=25.11dev
FROM ${TRITONSDK_BASE_IMAGE} AS sdk

FROM ${BASE_IMAGE}
ARG MODEL_ANALYZER_VERSION
ARG MODEL_ANALYZER_CONTAINER_VERSION
ARG BASE_IMAGE
ARG TRITONSDK_BASE_IMAGE

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -qq && apt install -y docker.io

# Install tritonclient
COPY --from=sdk /workspace/install/python /tmp/tritonclient
RUN find /tmp/tritonclient -maxdepth 1 -type f -name \
    "tritonclient-*-manylinux*.whl" | xargs printf -- '%s[all]' | \
    xargs pip3 install --upgrade && rm -rf /tmp/tritonclient/

WORKDIR /opt/triton-model-analyzer

COPY . .
RUN chmod +x /opt/triton-model-analyzer/nvidia_entrypoint.sh

RUN python3 -m pip install \
        coverage \
        mkdocs \
        mkdocs-htmlproofer-plugin==0.10.3 \
        mypy \
        nvidia-pyindex \
        types-protobuf \
        types-PyYAML \
        types-requests \
        yapf==0.32.0

RUN python3 setup.py bdist_wheel \
    && cd dist \
    && python3 -m pip install triton*model*analyzer*.whl
ENTRYPOINT ["/opt/triton-model-analyzer/nvidia_entrypoint.sh"]
ENV MODEL_ANALYZER_VERSION=${MODEL_ANALYZER_VERSION}
ENV MODEL_ANALYZER_CONTAINER_VERSION=${MODEL_ANALYZER_CONTAINER_VERSION}
ENV TRITON_SERVER_SDK_CONTAINER_IMAGE_NAME=${TRITONSDK_BASE_IMAGE}
ENV TRITON_SERVER_CONTAINER_IMAGE_NAME=${BASE_IMAGE}

