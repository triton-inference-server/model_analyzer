# Copyright 2020, NVIDIA CORPORATION.
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
