#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

set -e
cat <<EOF
=============================
=== Triton Model Analyzer ===
=============================
NVIDIA Release ${MODEL_ANALYZER_CONTAINER_VERSION} (build ${NVIDIA_BUILD_ID})
Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.
This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
EOF

if [[ "$(find -L /usr -name libcuda.so.1 | grep -v "compat") " == " " || "$(ls /dev/nvidiactl 2>/dev/null) " == " " ]]; then
  echo
  echo "WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available."
  echo "   Use Docker with NVIDIA Container Toolkit to start this container; see"
  echo "   https://github.com/NVIDIA/nvidia-docker."
  ln -s `find / -name libnvidia-ml.so -print -quit` /opt/tritonserver/lib/libnvidia-ml.so.1
  export TRITON_SERVER_CPU_ONLY=1
else
  ( /usr/local/bin/checkSMVER.sh )
  DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
  if [[ ! "$DRIVER_VERSION" =~ ^[0-9]*.[0-9]*(.[0-9]*)?$ ]]; then
    echo "Failed to detect NVIDIA driver version."
  elif [[ "${DRIVER_VERSION%%.*}" -lt "${CUDA_DRIVER_VERSION%%.*}" ]]; then
    if [[ "${_CUDA_COMPAT_STATUS}" == "CUDA Driver OK" ]]; then
      echo
      echo "NOTE: Legacy NVIDIA Driver detected.  Compatibility mode ENABLED."
    else
      echo
      echo "ERROR: This container was built for NVIDIA Driver Release ${CUDA_DRIVER_VERSION%.*} or later, but"
      echo "       version ${DRIVER_VERSION} was detected and compatibility mode is UNAVAILABLE."
      echo
      echo "       [[${_CUDA_COMPAT_STATUS}]]"
      sleep 2
    fi
  fi
fi

if ! cat /proc/cpuinfo | grep flags | sort -u | grep avx >& /dev/null; then
  echo
  echo "ERROR: This container was built for CPUs supporting at least the AVX instruction set, but"
  echo "       the CPU detected was $(cat /proc/cpuinfo |grep "model name" | sed 's/^.*: //' | sort -u), which does not report"
  echo "       support for AVX.  An Illegal Instrution exception at runtime is likely to result."
  echo "       See https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX ."
  sleep 2
fi

echo

if [[ $# -eq 0 ]]; then
  exec "/bin/bash"
else
  exec "$@"
fi
