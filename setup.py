# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys

from setuptools import find_packages
from setuptools import setup
from itertools import chain


def version(filename='VERSION'):
    with open(os.path.join(filename)) as f:
        project_version = f.read()
    return project_version


project_version = version()
this_directory = os.path.abspath(os.path.dirname(__file__))


def req_file(filename):
    with open(os.path.join(filename)) as f:
        content = f.readlines()
    return [x.strip() for x in content if not x.startswith("#")]


install_requires = req_file("requirements.txt")

data_files = [
    ("", ["LICENSE"]),
]

setup(
    name='model-analyzer',
    version=project_version,
    author='NVIDIA Inc.',
    author_email='sw-dl-triton@nvidia.com',
    description=
    "The Model Analyzer is a tool to analyze the runtime performance of one or more models on the Triton Inference Server",
    license='BSD',
    url='https://developer.nvidia.com/nvidia-triton-inference-server',
    keywords=[
        'triton', 'tensorrt', 'inference', 'server', 'service', 'analyzer',
        'nvidia'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Utilities',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: Linux',
    ],
    entry_points={
        'console_scripts': ['model-analyzer = model_analyzer.entrypoint:main']
    },
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
    data_files=data_files,
)
