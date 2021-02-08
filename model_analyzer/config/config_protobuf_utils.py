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

from google.protobuf.descriptor import FieldDescriptor
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException


def is_protobuf_type_primitive(protobuf_type):
    """
    Check whether the protobuf type is primitive (i.e. not Message).

    Parameters
    ----------
    protobuf_type : int
        Protobuf type

    Returns
    -------
    bool
        True if the protobuf type is a primitive.
    """

    if protobuf_type == FieldDescriptor.TYPE_BOOL:
        return True
    elif protobuf_type == FieldDescriptor.TYPE_DOUBLE:
        return True
    elif protobuf_type == FieldDescriptor.TYPE_FLOAT:
        return True
    elif protobuf_type == FieldDescriptor.TYPE_INT32:
        return True
    elif protobuf_type == FieldDescriptor.TYPE_INT64:
        return True
    elif protobuf_type == FieldDescriptor.TYPE_STRING:
        return True
    elif protobuf_type == FieldDescriptor.TYPE_UINT32:
        return True
    elif protobuf_type == FieldDescriptor.TYPE_UINT64:
        return True
    else:
        return False


def protobuf_to_config_type(protobuf_type):
    """
    Map the protobuf type to the Python types.

    Parameters
    ----------
    protobuf_type : int
        The protobuf type to be mapped.

    Returns
    -------
    type or bool
        The equivalent Python type for the protobuf type. If the type is not
        supported, it will return False.

    Raises
    ------
    TritonModelAnalyzerException
        If the protobuf_type is not a primitive type, this exception will be
        raised.
    """

    if not is_protobuf_type_primitive(protobuf_type):
        raise TritonModelAnalyzerException()

    # TODO: Is using int for INT32 and INT64 ok? Similarly for UINT32, UINT64,
    # FLOAT, DOUBLE.
    if protobuf_type == FieldDescriptor.TYPE_BOOL:
        return bool
    elif protobuf_type == FieldDescriptor.TYPE_DOUBLE:
        return float
    elif protobuf_type == FieldDescriptor.TYPE_FLOAT:
        return float
    elif protobuf_type == FieldDescriptor.TYPE_INT32:
        return int
    elif protobuf_type == FieldDescriptor.TYPE_INT64:
        return int
    elif protobuf_type == FieldDescriptor.TYPE_STRING:
        return str
    elif protobuf_type == FieldDescriptor.TYPE_UINT32:
        return int
    elif protobuf_type == FieldDescriptor.TYPE_UINT64:
        return int
    else:
        return False
