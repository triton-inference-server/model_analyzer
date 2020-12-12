# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:20.11-py3
ARG TRITONSDK_BASE_IMAGE=nvcr.io/nvidia/tritonserver:20.11-py3-clientsdk

ARG MODEL_ANALYZER_VERSION=1.0.0dev
ARG MODEL_ANALYZER_CONTAINER_VERSION=20.12dev

FROM ${TRITONSDK_BASE_IMAGE} as sdk

FROM $BASE_IMAGE
ARG MODEL_ANALYZER_VERSION
ARG MODEL_ANALYZER_CONTAINER_VERSION

# DCGM version to install for Model Analyzer
ENV DCGM_VERSION=2.0.13

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# Install tritonclient from the SDK container
COPY --from=sdk /workspace/install/python /tmp/tritonclient
RUN find /tmp/tritonclient -maxdepth 1 -type f -name \
    "tritonclient-*-manylinux1_x86_64.whl" | xargs printf -- '%s[all]' | \
    xargs pip3 install --upgrade && rm -rf /tmp/tritonclient/

RUN mkdir -p /opt/triton-model-analyzer

# Install DCGM
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    dpkg -i datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb

WORKDIR /opt/triton-model-analyzer
RUN rm -fr *
COPY --from=sdk /usr/local/bin/perf_analyzer .
COPY . .
RUN chmod +x build_wheel.sh && \
    ./build_wheel.sh perf_analyzer true && \
    rm -f perf_analyzer
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install wheels/nvidia_triton_model_analyzer-*-manylinux1_x86_64.whl

ENTRYPOINT []
ENV MODEL_ANALYZER_VERSION ${MODEL_ANALYZER_VERSION}
ENV NVIDIA_MODEL_ANALYZER_VERSION ${MODEL_ANALYZER_CONTAINER_VERSION}
