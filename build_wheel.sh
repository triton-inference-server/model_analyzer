#!/bin/bash
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

PYTHON=${PYTHON:=`which python3`}

if [[ "$1" == "-h" || "$1" == "--help" ]] ; then
    echo "usage: $0 <perf-analyzer-binary> <linux-specific>"
    exit 1
fi

if [[ ! -f "VERSION" ]]; then
    echo "Could not find VERSION"
    exit 1
fi

if [[ ! -f "LICENSE" ]]; then
    echo "Could not find LICENSE"
    exit 1
fi

if [[ -z "$1" ]]; then
    echo "Path to perf_analyzer binary not provided. Checking PATH..."
    if [[ -z "$(which perf_analyzer)" ]]; then
        echo "Could not find perf_analyzer binary"
        exit 1
    else
        PERF_ANALYZER_PATH="$(which perf_analyzer)"
    fi
elif [[ ! -f "$1" ]]; then
    echo "Could not find perf_analyzer binary"
    exit 1
else
    PERF_ANALYZER_PATH="${1}"
fi

WHLDIR="`pwd`/wheels"
mkdir -p ${WHLDIR}

# Copy required files into WHEELDIR temporarily
cp $PERF_ANALYZER_PATH "${WHLDIR}"
cp VERSION "${WHLDIR}"
cp LICENSE "${WHLDIR}"
cp requirements.txt "${WHLDIR}"

# Set platform and build wheel
echo $(date) : "=== Building wheel"
if [[ -z "$2" || "$2" = true ]]; then
    PLATFORM=`uname -m`
    if [ "$PLATFORM" = "aarch64" ] ; then
        PLATFORM_NAME="linux_aarch64"
    else
        PLATFORM_NAME="manylinux1_x86_64"
    fi
    ${PYTHON} setup.py  bdist_wheel --plat-name $PLATFORM_NAME --dependency-dir $WHLDIR
else
    ${PYTHON} setup.py bdist_wheel --dependency-dir $WHLDIR
fi
rm -f $WHLDIR/* && cp dist/* $WHLDIR
rm -rf build dist triton_model_analyzer.egg-info
touch ${WHLDIR}/stamp.whl
echo $(date) : "=== Output wheel file is in: ${WHLDIR}"
