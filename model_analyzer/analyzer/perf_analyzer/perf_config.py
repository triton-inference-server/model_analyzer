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

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class PerfAnalyzerConfig:
    """
    A config class to set arguments to the perf_analyzer.
    An argument set to None will use the perf_analyzer's default.
    """
    def __init__(self):
        self._args = {
            'async': None,
            'sync': None,
            'measurement-interval': None,
            'concurrency-range': None,
            'request-rate-range': None,
            'request-distribution': None,
            'request-intervals': None,
            'binary-search': None,
            'num-of-sequence': None,
            'latency-threshold': None,
            'max-threads': None,
            'stability-percentage': None,
            'max-trials': None,
            'percentile': None,
            'input-data': None,
            'shared-memory': None,
            'output-shared-memory-size': None,
            'shape': None,
            'sequence-length': None,
            'string-length': None,
            'string-data': None,
        }

        self._options = {
            '-m': None,
            '-x': None,
            '-b': None,
            '-u': None,
            '-i': None,
            '-f': None,
            '-H': None
        }

        self._verbose = {'-v': None, '-v -v': None}

        self._input_to_options = {
            'model-name': '-m',
            'model-version': '-x',
            'batch-size': '-b',
            'url': '-u',
            'protocol': '-i',
            'latency-report-file': '-f',
            'streaming': '-H'
        }

        self._input_to_verbose = {'verbose': '-v', 'extra-verbose': '-v -v'}

    def to_cli_string(self):
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
        args = [f'{k} {v}' for k, v in self._options.items() if v]
        args += [k for k, v in self._verbose.items() if v]
        args += [f'--{k}={v}' for k, v in self._args.items() if v]

        return ' '.join(args)

    def __getitem__(self, key):
        """
        Gets an arguments value in config

        Parameters
        ----------
        key : str
            The name of the argument to the tritonserver

        Returns
        -------
            The value that the argument is set to in this config

        Raises
        ------
        TritonModelAnalyzerException
            If argument not found in the config
        """
        if key in self._args:
            return self._args[key]
        elif key in self._input_to_options:
            self._options[self._input_to_options[key]] = value
        elif key in self._input_to_verbose:
            self._verbose[self._input_to_verbose[key]] = value
        else:
            raise TritonModelAnalyzerException(
                f"'{key}' Key not found in config")

    def __setitem__(self, key, value):
        """
        Sets an arguments value in config
        after checking if defined/supported.

        Parameters
        ----------
        key : str
            The name of the argument to the tritonserver
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
        elif key in self._input_to_options:
            self._options[self._input_to_options[key]] = value
        elif key in self._verbose:
            self._verbose[self._input_to_verbose[key]] = value
        else:
            raise TritonModelAnalyzerException(
                f"The argument '{key}' to the perf_analyzer "
                "is not supported by the model analyzer.")
