# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.06-py3
ARG TRITONSDK_BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.06-py3-sdk

ARG MODEL_ANALYZER_VERSION=1.19.0dev
ARG MODEL_ANALYZER_CONTAINER_VERSION=22.08dev

FROM ${TRITONSDK_BASE_IMAGE} as sdk

FROM $BASE_IMAGE
ARG MODEL_ANALYZER_VERSION
ARG MODEL_ANALYZER_CONTAINER_VERSION
ARG BASE_IMAGE
ARG TRITONSDK_BASE_IMAGE

# DCGM version to install for Model Analyzer
ENV DCGM_VERSION=2.2.9

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3-dev

RUN mkdir -p /opt/triton-model-analyzer

# Install architecture-specific components
    # Install DCGM

RUN [ "$(uname -m)" != "x86_64" ] && arch="sbsa" || arch="x86_64" && \
    curl -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/$arch/cuda-keyring_1.0-1_all.deb && \
    apt-get install /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb && \
    apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    apt-get install -y datacenter-gpu-manager=1:${DCGM_VERSION}; 
    
    # Install Docker
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
   apt-get update && apt-get install -y docker-ce-cli

# Install tritonclient
COPY --from=sdk /workspace/install/python /tmp/tritonclient
RUN find /tmp/tritonclient -maxdepth 1 -type f -name \
    "tritonclient-*-manylinux*.whl" | xargs printf -- '%s[all]' | \
    xargs pip3 install --upgrade && rm -rf /tmp/tritonclient/

WORKDIR /opt/triton-model-analyzer
RUN rm -fr *
COPY --from=sdk /usr/local/bin/perf_analyzer .
RUN chmod +x ./perf_analyzer

COPY . .
RUN chmod +x /opt/triton-model-analyzer/nvidia_entrypoint.sh
RUN chmod +x build_wheel.sh && \
    ./build_wheel.sh perf_analyzer true && \
    rm -f perf_analyzer
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install nvidia-pyindex && \
    python3 -m pip install wheels/triton_model_analyzer-*-manylinux*.whl
#Other pip packages
RUN python3 -m pip install \
    coverage

RUN apt-get install -y wkhtmltopdf

ENTRYPOINT ["/opt/triton-model-analyzer/nvidia_entrypoint.sh"]
ENV MODEL_ANALYZER_VERSION ${MODEL_ANALYZER_VERSION}
ENV MODEL_ANALYZER_CONTAINER_VERSION ${MODEL_ANALYZER_CONTAINER_VERSION}
ENV TRITON_SERVER_SDK_CONTAINER_IMAGE_NAME ${TRITONSDK_BASE_IMAGE}
ENV TRITON_SERVER_CONTAINER_IMAGE_NAME ${BASE_IMAGE}

