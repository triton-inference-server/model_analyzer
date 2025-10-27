#!/usr/bin/env bash
# Copyright 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -euo pipefail

# --- Helpers ---------------------------------------------------------------

first_lib() {
  # Prefer ldconfig (fast/quiet), fall back to a bounded dir scan
  local pat="$1"
  local p
  p="$(ldconfig -p 2>/dev/null | awk -v re="$pat" '$0 ~ re {print $4; exit}')" || true
  if [[ -z "${p:-}" ]]; then
    for d in /usr/lib /usr/lib64 /lib /lib64 /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu; do
      [[ -d "$d" ]] || continue
      p="$(printf '%s\n' "$d"/lib*.so* 2>/dev/null | awk -v re="$pat" '$0 ~ re {print; exit}')" || true
      [[ -n "${p:-}" ]] && break
    done
  fi
  [[ -n "${p:-}" ]] && printf '%s\n' "$p"
}

has_nvidia_driver() {
  [[ -e /dev/nvidiactl ]] || return 1
  [[ -n "$(first_lib 'libcuda\.so\.1($| )')" ]]
}

# --- GPU presence / compatibility -----------------------------------------

if ! has_nvidia_driver; then
  echo
  echo "WARNING: The NVIDIA Driver was not detected. GPU functionality will not be available."
  echo "   Use Docker with NVIDIA Container Toolkit: https://github.com/NVIDIA/nvidia-docker"
  # Some Triton paths expect libnvidia-ml.so.1; create a symlink if we can find any variant.
  if ml="$(first_lib 'libnvidia-ml\.so(\.|$)')" ; then
    install -d /opt/tritonserver/lib
    ln -sf "$ml" /opt/tritonserver/lib/libnvidia-ml.so.1 || true
  fi
  export TRITON_SERVER_CPU_ONLY=1
else
  DRIVER_VERSION="$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)"
  if [[ -z "${DRIVER_VERSION}" || ! "$DRIVER_VERSION" =~ ^[0-9]+(\.[0-9]+){0,2}$ ]]; then
    echo "Failed to detect NVIDIA driver version."
  elif [[ -n "${CUDA_DRIVER_VERSION:-}" ]] && [[ "${DRIVER_VERSION%%.*}" -lt "${CUDA_DRIVER_VERSION%%.*}" ]]; then
    if [[ "${_CUDA_COMPAT_STATUS:-}" == "CUDA Driver OK" ]]; then
      echo
      echo "NOTE: Legacy NVIDIA Driver detected. Compatibility mode ENABLED."
    else
      echo
      echo "ERROR: This container was built for NVIDIA Driver Release ${CUDA_DRIVER_VERSION%.*} or later,"
      echo "       but version ${DRIVER_VERSION} was detected and compatibility mode is UNAVAILABLE."
      echo
      echo "       [[${_CUDA_COMPAT_STATUS:-unset}]]"
      sleep 2
    fi
  fi
fi

# --- CPU AVX advisory ------------------------------------------------------

if ! grep -qm1 ' avx' /proc/cpuinfo; then
  echo
  echo "ERROR: This container was built for CPUs supporting at least the AVX instruction set,"
  echo "       but the detected CPU ($(grep -m1 'model name' /proc/cpuinfo | sed 's/^.*: //')) does not report AVX."
  echo "       An Illegal Instruction exception at runtime is likely to result."
  echo "       See https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX ."
  sleep 2
fi

echo

# --- Hand off --------------------------------------------------------------

if [[ $# -eq 0 ]]; then
  exec /bin/bash
else
  exec "$@"
fi