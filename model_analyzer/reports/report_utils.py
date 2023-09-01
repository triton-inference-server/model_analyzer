#!/usr/bin/env python3

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def truncate_model_config_name(model_config_name):
    """
    Truncates the model configuration name if its length exceeds the threshold length.
    ex: long_model_name_config_4  -->   long_mod..._config_4
    Parameters
    ----------
    model_config_name: string
    Returns
    -------
    string
        The truncated model configuration name,
        or the original name if it is shorter than the threshold length.
    """
    max_model_config_name_len = 35

    if len(model_config_name) > max_model_config_name_len:
        config_name = model_config_name[model_config_name.rfind("config_") :]

        return (
            model_config_name[: (max_model_config_name_len - len(config_name) - 3)]
            + "..."
            + config_name
        )

    return model_config_name
