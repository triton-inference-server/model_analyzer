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

import os
import sys

from setuptools import find_packages
from setuptools import setup

if "--dependency-dir" in sys.argv:
    idx = sys.argv.index("--dependency-dir")
    DEPENDENCY_DIR = sys.argv[idx + 1]
    sys.argv.pop(idx + 1)
    sys.argv.pop(idx)
else:
    DEPENDENCY_DIR = '.'

if "--plat-name" in sys.argv:
    PLATFORM_FLAG = sys.argv[sys.argv.index("--plat-name") + 1]
else:
    PLATFORM_FLAG = 'any'


def version(filename='VERSION'):
    with open(os.path.join(filename)) as f:
        project_version = f.read()
    return project_version


def req_file(filename):
    with open(os.path.join(filename)) as f:
        content = f.readlines()
    return [x.strip() for x in content if not x.startswith("#")]


project_version = version()
install_requires = req_file("requirements.txt")

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            pyver, abi, plat = 'py3', 'none', PLATFORM_FLAG
            return pyver, abi, plat
except ImportError:
    bdist_wheel = None

data_files = [
    ("", [os.path.join(DEPENDENCY_DIR, "LICENSE")]),
]

if PLATFORM_FLAG != 'any':
    data_files += [("bin", [os.path.join(DEPENDENCY_DIR, "perf_analyzer")])]

setup(
    name='triton-model-analyzer',
    version=project_version,
    author='NVIDIA Inc.',
    author_email='sw-dl-triton@nvidia.com',
    description=
    "Triton Model Analyzer is a tool to analyze the runtime performance of one or more models on the Triton Inference Server",
    long_description=
    """See the Model Analyzer's [installation documentation](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/install.md#using-pip3) """
    """for package details. The [quick start](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/quick_start.md) documentation """
    """describes how to get started with profiling and analysis using Triton Model Analyzer.""",
    long_description_content_type='text/markdown',
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
        'Operating System :: POSIX :: Linux',
    ],
    entry_points={
        'console_scripts': ['model-analyzer = model_analyzer.entrypoint:main']
    },
    install_requires=install_requires,
    dependency_links=['https://pypi.ngc.nvidia.com/tritonclient'],
    packages=find_packages(exclude=("tests", )),
    zip_safe=False,
    data_files=data_files,
)
