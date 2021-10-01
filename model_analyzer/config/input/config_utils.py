# Copyright 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .config_primitive import ConfigPrimitive
from model_analyzer.constants import CONFIG_PARSER_FAILURE, CONFIG_PARSER_SUCCESS
from .config_status import ConfigStatus

import os
import shutil

##############
# Validators #
##############


def parent_path_validator(path):
    """
    Perform a check on parent directory. Used when the file at
    'path' can be created by Model Analyzer in order to validate
    the parent directory of the file.

    Parameters
    ----------
    path: str
        Absolute path to a file or directory

    Returns
    -------
    ConfigStatus
    """

    abspath = os.path.abspath(path)

    if os.path.exists(os.path.dirname(abspath)):
        return ConfigStatus(status=CONFIG_PARSER_SUCCESS)
    else:
        return ConfigStatus(
            status=CONFIG_PARSER_FAILURE,
            message=
            f"Either the parent directory for '{path}' does not exist, or Model Analyzer "
            "does not have permissions to execute os.stat on this path.")


def binary_path_validator(path):
    """
    Used when path must refer to a valid binary
    such as for tritonserver or perf_analyzer.

    Parameters
    ----------
    path: str
        name of a binary if on PATH, or
        absolute path to a binary.

    Returns
    -------
    ConfigStatus
    """

    if shutil.which(path):
        return ConfigStatus(status=CONFIG_PARSER_SUCCESS)
    else:
        return ConfigStatus(
            status=CONFIG_PARSER_FAILURE,
            message=
            f"Either the binary '{path}' is not on the PATH, or Model Analyzer does "
            "not have permissions to execute os.stat on this path.")


def file_path_validator(path):
    """
    Perform some basic checks on strings passed in as paths.
    Used when the path is expected to exist, and Model Analyzer
    does not create the file at that path.

    Parameters
    ----------
    path: str
        Absolute path to a file or directory

    Returns
    -------
    ConfigStatus
    """

    abspath = os.path.abspath(path)

    if os.path.exists(abspath):
        return ConfigStatus(status=CONFIG_PARSER_SUCCESS)
    else:
        return ConfigStatus(
            status=CONFIG_PARSER_FAILURE,
            message=
            f"Either '{path}' is a nonexistent path, or Model Analyzer does "
            "not have permissions to execute os.stat on this path")


##################
# Output mappers #
##################


def objective_list_output_mapper(objectives):
    """
    Takes a list of objectives and maps them
    into a dict
    """

    output_dict = {}
    for objective in objectives:
        value = ConfigPrimitive(type_=int)
        value.set_value(10)
        output_dict[objective] = value
    return output_dict
