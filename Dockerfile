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

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:21.10-py3
ARG TRITONSDK_BASE_IMAGE=nvcr.io/nvidia/tritonserver:21.10-py3-sdk

ARG MODEL_ANALYZER_VERSION=1.10.0dev
ARG MODEL_ANALYZER_CONTAINER_VERSION=21.11dev

FROM ${TRITONSDK_BASE_IMAGE} as sdk

FROM $BASE_IMAGE
ARG MODEL_ANALYZER_VERSION
ARG MODEL_ANALYZER_CONTAINER_VERSION

# DCGM version to install for Model Analyzer
ENV DCGM_VERSION=2.2.9

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3-dev

RUN mkdir -p /opt/triton-model-analyzer

# Install architecture-specific components
ARG TARGETARCH
RUN \
    # Set TARGETARCH variable if in a non-docker buildx build
    if [ "${TARGETARCH}" = "" ] ; then \
      if [ `uname -m` = "x86_64" ] ; then TARGETARCH="amd64" ; \
      elif [ `uname -m` = "aarch64" ] ; then TARGETARCH="arm64" ; \
      fi ; \
    fi ; \

    # Exit if TARGETARCH variable is an invalid value
    if [ "${TARGETARCH}" != "amd64" ] && [ "${TARGETARCH}" != "arm64" ] ; then \
      echo "invalid architecture: $(uname -m)" ; exit 1 ; \
    fi ; \

    # Install libgfortran5 for arm64 version of docker image
    if [ "${TARGETARCH}" = "arm64" ] ; then \
      apt update ; apt install -y libgfortran5 ; \
    fi ; \

    # Set ARCH_DIR variable for correct architecture-specific DCGM installation
    if [ "${TARGETARCH}" = "amd64" ] ; then ARCH_DIR="x86_64" ; \
    elif [ "${TARGETARCH}" = "arm64" ] ; then ARCH_DIR="sbsa" ; \
    fi ; \

    # Install DCGM
    apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${ARCH_DIR}/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${ARCH_DIR}/7fa2af80.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${ARCH_DIR}/ /" && \
    apt-get install -y datacenter-gpu-manager=1:${DCGM_VERSION}

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

RUN apt-get install -y wkhtmltopdf

ENTRYPOINT ["/opt/triton-model-analyzer/nvidia_entrypoint.sh"]
ENV MODEL_ANALYZER_VERSION ${MODEL_ANALYZER_VERSION}
ENV MODEL_ANALYZER_CONTAINER_VERSION ${MODEL_ANALYZER_CONTAINER_VERSION}

