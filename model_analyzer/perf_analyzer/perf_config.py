# Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from model_analyzer.config.input.config_defaults import DEFAULT_MEASUREMENT_MODE
from model_analyzer.constants import SECONDS_TO_MILLISECONDS_MULTIPLIER


class PerfAnalyzerConfig:
    """
    A config class to set arguments to the perf_analyzer.
    An argument set to None will use the perf_analyzer's default.
    """

    perf_analyzer_args = [
        'service-kind', 'model-signature-name', 'async', 'sync',
        'measurement-interval', 'concurrency-range', 'request-rate-range',
        'request-distribution', 'request-intervals', 'binary-search',
        'num-of-sequences', 'latency-threshold', 'max-threads',
        'stability-percentage', 'max-trials', 'percentile', 'input-data',
        'shared-memory', 'output-shared-memory-size', 'shape',
        'sequence-length', 'sequence-id-range', 'string-length', 'string-data',
        'measurement-mode', 'measurement-request-count', 'streaming',
        'grpc-compression-algorithm', 'triton-server-directory',
        'model-repository', 'ssl-grpc-use-ssl',
        'ssl-grpc-root-certifications-file', 'ssl-grpc-private-key-file',
        'ssl-grpc-certificate-chain-file', 'ssl-https-verify-peer',
        'ssl-https-verify-host', 'ssl-https-ca-certificates-file',
        'ssl-https-client-certificate-type',
        'ssl-https-client-certificate-file', 'ssl-https-private-key-type',
        'ssl-https-private-key-file', 'collect-metrics', 'metrics-url',
        'metrics-interval', 'bls-composing-models'
    ]

    input_to_options = [
        'model-name', 'model-version', 'batch-size', 'url', 'protocol',
        'latency-report-file', 'http-header'
    ]

    input_to_verbose = ['verbose', 'extra-verbose', 'verbose-csv']

    additive_args = ['input-data', 'shape']

    boolean_args = [
        'streaming', 'async', 'sync', 'binary-search', 'ssl-grpc-use-ssl',
        'collect-metrics'
    ]

    def __init__(self):
        """
        Construct a PerfAnalyzerConfig
        """

        self._args = {k: None for k in self.perf_analyzer_args}

        self._options = {
            '-m': None,
            '-x': None,
            '-b': None,
            '-u': None,
            '-i': None,
            '-f': None,
            '-H': None
        }
        self._verbose = {'-v': None, '-v -v': None, '--verbose-csv': None}

        self._input_to_options = {
            'model-name': '-m',
            'model-version': '-x',
            'batch-size': '-b',
            'url': '-u',
            'protocol': '-i',
            'latency-report-file': '-f',
            'http-header': '-H'
        }

        self._input_to_verbose = {
            'verbose': '-v',
            'extra-verbose': '-v -v',
            'verbose-csv': '--verbose-csv'
        }

        self._additive_args = {
            (self._input_to_options[k] if k in self._input_to_options else k):
            None for k in self.additive_args
        }

    @classmethod
    def allowed_keys(cls):
        """
        Returns
        -------
        list of str
            The keys that are allowed to be
            passed into perf_analyzer
        """

        return cls.perf_analyzer_args + cls.input_to_options + cls.input_to_verbose

    @classmethod
    def additive_keys(cls):
        """
        Returns
        -------
        list of str
            The keys, within allowed_keys, that are additive
        """

        return cls.additive_args[:]

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

    def update_config_from_profile_config(self, model_name, profile_config):
        """
        Set common values based on the input profile config

        Parameters
        ----------
        model_name: str
            The name of the model
        profile_config: ConfigCommandProfile
            The configuration of model analyzer for the profile step
        """

        params = {
            'model-name': model_name,
            'latency-report-file': model_name + "-results.csv",
            'measurement-mode': DEFAULT_MEASUREMENT_MODE,
            'verbose-csv': '--verbose-csv'
        }

        if profile_config.triton_launch_mode == 'c_api':
            params.update({
                'service-kind': 'triton_c_api',
                'triton-server-directory': profile_config.triton_install_path,
                'model-repository': profile_config.output_model_repository_path
            })
        else:
            if profile_config.client_protocol == 'http':
                url = profile_config.triton_http_endpoint
            else:
                url = profile_config.triton_grpc_endpoint

            params.update({
                'protocol': profile_config.client_protocol,
                'url': url
            })

            metrics_interval = profile_config.monitoring_interval * SECONDS_TO_MILLISECONDS_MULTIPLIER
            params.update({
                'collect-metrics': 'True',
                'metrics-url': profile_config.triton_metrics_url,
                'metrics-interval': metrics_interval
            })

        if profile_config.bls_composing_models:
            bls_composing_model_names = ','.join([
                bls_composing_model.model_name()
                for bls_composing_model in profile_config.bls_composing_models
            ])
            params.update({'bls-composing-models': bls_composing_model_names})

        self.update_config(params)

    @classmethod
    def from_dict(cls, perf_config_dict):
        perf_config = PerfAnalyzerConfig()
        for key in [
                '_args', '_options', '_verbose', '_input_to_verbose',
                '_input_to_options'
        ]:
            if key in perf_config_dict:
                setattr(perf_config, key, perf_config_dict[key])
        return perf_config

    def representation(self):
        """
        Returns
        -------
        str
            a string representation of the PA config 
            that removes values which can vary between
            runs, but should be ignored when determining
            if a previous (checkpointed) run can be used
        """
        cli_string = self.to_cli_string()
        cli_string = PerfAnalyzerConfig.remove_url_from_cli_string(cli_string)
        cli_string = PerfAnalyzerConfig.remove_mrc_from_cli_string(cli_string)

        return cli_string

    def extract_model_specific_parameters(self):
        """
        Returns a dictionary of the parameters (options+args) that can change
        between the models (in a ModelRunConfig) within a RunConfig
        Returns
        -------
        dict
          A dictionary of parameters and their values
        """

        return {
            'batch-size': self._options['-b'],
            'concurrency-range': self._args['concurrency-range'],
            'request-rate-range': self._args['request-rate-range']
        }

    @classmethod
    def remove_url_from_cli_string(cls, cli_string):
        """
        utility function strips the url from a cli
        string representation

        Parameters
        ----------
        cli_string : str
            The cli string representation
        """

        perf_str_tokens = cli_string.split(' ')

        try:
            url_index = perf_str_tokens.index('-u')
            # remove -u and the element that comes after it
            perf_str_tokens.pop(url_index)
            perf_str_tokens.pop(url_index)
        except ValueError:
            pass

        return ' '.join(perf_str_tokens)

    @classmethod
    def remove_mrc_from_cli_string(cls, cli_string):
        """
        utility function strips the measruement request count
        from a cli string representation

        Parameters
        ----------
        cli_string : str
            The cli string representation
        """

        perf_str_tokens = cli_string.split(' ')

        mrc_index = [
            i for i, s in enumerate(perf_str_tokens)
            if '--measurement-request-count' in s
        ]

        if mrc_index:
            perf_str_tokens.pop(mrc_index[0])

        return ' '.join(perf_str_tokens)

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
        args = []
        args.extend(self._parse_short_options())
        args.extend(self._parse_verbose_options())
        args.extend(self._parse_long_options())

        return ' '.join(args)

    def _parse_short_options(self):
        """
        Parse the perf analyzer single dash options
        """
        temp_args = []
        for key, value in self._options.items():
            if value:
                if key in self._additive_args:
                    for additive_value in value:
                        temp_args.append(f'{key} {additive_value}')
                else:
                    temp_args.append(f'{key} {value}')
        return temp_args

    def _parse_verbose_options(self):
        """
        Add the verbose flags to the args cli list
        """
        return [k for k, v in self._verbose.items() if v]

    def _parse_long_options(self):
        """
        Parse the perf analyzer long args 
        """
        temp_args = []
        for key, value in self._args.items():
            if key in self.boolean_args:
                temp_args = self._parse_boolean_args(key, value, temp_args)
            elif value:
                if key in self._additive_args:
                    if type(value) is list:
                        for additive_value in value:
                            temp_args.append(f'--{key}={additive_value}')
                    elif type(value) is str:
                        temp_args.append(f'--{key}={value}')
                    else:
                        raise TritonModelAnalyzerException(
                            f"Unexpected type {type(value)} for perf_analyzer_flag {key}."
                        )
                else:
                    temp_args.append(f'--{key}={value}')
        return temp_args

    def _parse_boolean_args(self, key, value, temp_args):
        """
        Parse perf analyzer long args that should not add a value to the cli string
        """
        assert type(value) in [
            str, type(None)
        ], f"Data type for arg {key} must be a (boolean) string instead of {type(value)}"
        if value != None and value.lower() == "true":
            temp_args.append(f'--{key}')
        return temp_args

    def __getitem__(self, key):
        """
        Gets an arguments value in config

        Parameters
        ----------
        key : str
            The name of the argument to the perf config

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
        elif key in self._input_to_options:
            return self._options[self._input_to_options[key]]
        elif key in self._input_to_verbose:
            return self._verbose[self._input_to_verbose[key]]
        else:
            raise TritonModelAnalyzerException(
                f'Key {key} does not exist in perf_analyzer_flags.')

    def __setitem__(self, key, value):
        """
        Sets an arguments value in config
        after checking if defined/supported.

        Parameters
        ----------
        key : str
            The name of the argument in perf_analyzer
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
        elif key in self._input_to_verbose:
            self._verbose[self._input_to_verbose[key]] = value
        else:
            raise TritonModelAnalyzerException(
                f"The argument '{key}' to the perf_analyzer "
                "is not supported by the model analyzer.")

    def __contains__(self, key):
        """
        Returns
        -------
        True if key is in perf_config i.e. the key is a
        perf config argument
        """

        return key in PerfAnalyzerConfig.allowed_keys()
