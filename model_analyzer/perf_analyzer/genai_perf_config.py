#!/usr/bin/env python3

# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class GenaiPerfConfig:
    """
    A config class to set arguments to the genai_perf.
    An argument set to None will use the genai_perf's default.
    """

    genai_perf_args = [
        "backend",
        "endpoint",
        "service-kind",
        "url",
        "expected-output-tokens",
        "input-dataset",
        "input-tokens-mean",
        "input-tokens-stddev",
        "input-type",
        "num-of-output-prompts",
        "random-seed",
        "streaming",
        "tokenizer",
    ]

    boolean_args = ["streaming"]

    def __init__(self):
        """
        Construct a GenaiPerfConfig
        """

        self._args = {k: None for k in self.genai_perf_args}

    @classmethod
    def allowed_keys(cls):
        """
        Returns
        -------
        list of str
            The keys that are allowed to be
            passed into perf_analyzer
        """

        return cls.genai_perf_args

    def update_config(self, params=None):
        """
        Allows setting values from a params dict

        Parameters
        ----------
        params: dict
            keys are allowed args to perf_analyzer
        """

        if params and type(params) is dict:
            for key in params:
                self[key] = params[key]

    @classmethod
    def from_dict(cls, genai_perf_config_dict):
        genai_perf_config = GenaiPerfConfig()
        for key in [
            "_args",
        ]:
            if key in genai_perf_config_dict:
                setattr(genai_perf_config, key, genai_perf_config_dict[key])
        return genai_perf_config

    def representation(self):
        """
        Returns
        -------
        str
            a string representation of the Genai Perf config
            that removes values which can vary between
            runs, but should be ignored when determining
            if a previous (checkpointed) run can be used
        """
        cli_string = self.to_cli_string()

        return cli_string

    def to_cli_string(self) -> str:
        """
        Utility function to convert a config into a
        string of arguments to the perf_analyzer with CLI.

        Returns
        -------
        str
            cli command string consisting of all arguments
            to the perf_analyzer set in the config, without
            the executable name.
        """

        # single dashed options, then verbose flags, then main args
        args = []
        args.extend(self._parse_options())

        return " ".join(args)

    def _parse_options(self):
        """
        Parse the genai perf args
        """
        temp_args = []
        for key, value in self._args.items():
            if key in self.boolean_args:
                temp_args = self._parse_boolean_args(key, value, temp_args)
            elif value:
                temp_args.append(f"--{key}={value}")
        return temp_args

    def _parse_boolean_args(self, key, value, temp_args):
        """
        Parse genai perf args that should not add a value to the cli string
        """
        assert type(value) in [
            str,
            type(None),
        ], f"Data type for arg {key} must be a (boolean) string instead of {type(value)}"
        if value != None and value.lower() == "true":
            temp_args.append(f"--{key}")
        return temp_args

    def __getitem__(self, key):
        """
        Gets an arguments value in config

        Parameters
        ----------
        key : str
            The name of the argument to the genai perf config

        Returns
        -------
        object
            The value that the argument is set to in this config

        Raises
        ------
        KeyError
            If argument not found in the config
        """

        if key in self._args:
            return self._args[key]
        else:
            raise TritonModelAnalyzerException(
                f"Key {key} does not exist in genai_perf_flags."
            )

    def __setitem__(self, key, value):
        """
        Sets an arguments value in config
        after checking if defined/supported.

        Parameters
        ----------
        key : str
            The name of the argument in genai_perf
        value : (any)
            The value to which the argument is being set

        Raises
        ------
        TritonModelAnalyzerException
            If key is unsupported or undefined in the
            config class
        """

        if key in self._args:
            self._args[key] = value
        else:
            raise TritonModelAnalyzerException(
                f"The argument '{key}' to the genai_perf "
                "is not supported by model analyzer."
            )

    def __contains__(self, key):
        """
        Returns
        -------
        True if key is in perf_config i.e. the key is a
        genai perf config argument
        """

        return key in GenaiPerfConfig.allowed_keys()
