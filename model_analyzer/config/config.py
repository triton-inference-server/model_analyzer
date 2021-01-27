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

import yaml
import logging
import os
from .config_field import ConfigField
from .config_primitive import ConfigPrimitive
from .config_list_string import ConfigListString
from .config_list_numeric import ConfigListNumeric
from .config_object import ConfigObject
from .config_list_generic import ConfigListGeneric
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class AnalyzerConfig:
    """
    Model Analyzer config object.
    """

    def __init__(self):
        """
        Create a new config.
        """

        self._fields = {}
        self._fill_config()

    def _add_config(self, config_field):
        """
        Add a new config field.

        Parameters
        ----------
        config_field : ConfigField
            Config field to be added

        Raises
        ------
        KeyError
            If the field already exists, it will raise this exception.
        """

        if config_field.name() not in self._fields:
            self._fields[config_field.name()] = config_field
        else:
            raise KeyError

    def _fill_config(self):
        self._add_config(
            ConfigField('model_repository',
                        flags=['--model-repository', '-m'],
                        field_types=[ConfigPrimitive(str, required=True)],
                        description='Model repository location'))

        constraints_scheme = ConfigObject(
            schema={
                'throughput': ConfigObject(schema={
                    'min': ConfigPrimitive(int),
                }),
                'latency': ConfigObject(schema={
                    'max': ConfigPrimitive(int),
                }),
                'gpu_memory': ConfigObject(schema={
                    'max': ConfigPrimitive(int),
                }),
            })
        model_object_constraint = ConfigObject(
            required=True,
            schema={
                # Any key is allowed, but the keys must follow the pattern below
                '*':
                ConfigObject(
                    schema={
                        'parameters':
                        ConfigObject(
                            schema={
                                'batch_sizes': ConfigListNumeric(type_=int),
                                'concurrency': ConfigListNumeric(type_=int)
                            }),
                        'objectives':
                        ConfigListString(),
                        'constraints':
                        constraints_scheme
                    })
            })
        self._add_config(
            ConfigField(
                'model_names',
                flags=['--model-names', '-n'],
                field_types=[
                    model_object_constraint,
                    ConfigListGeneric(
                        [model_object_constraint,
                         ConfigPrimitive(str)],
                        required=True),
                    ConfigListString(),
                ],
                description=
                'Comma-delimited list of the model names to be profiled'))

        self._add_config(
            ConfigField(
                'objectives',
                field_types=[ConfigListString()],
                description=
                'Model Analyzer uses the objectives described here to find the best configuration for each model.'
            ))

        self._add_config(
            ConfigField(
                'constraints',
                field_types=[constraints_scheme],
                description=
                'Constraints on the objectives specified in the "objectives" field of the config.'
            ))
        self._add_config(
            ConfigField(
                'batch_sizes',
                flags=['--batch-sizes', '-b'],
                field_types=[ConfigListNumeric(int)],
                default_value=1,
                description=
                'Comma-delimited list of batch sizes to use for the profiling')
        )
        self._add_config(
            ConfigField(
                'concurrency',
                flags=['-c', '--concurrency'],
                field_types=[ConfigListNumeric(int)],
                default_value=1,
                description=
                "Comma-delimited list of concurrency values or ranges <start:end:step>"
                " to be used during profiling"))
        self._add_config(
            ConfigField('export',
                        flags=['--export'],
                        field_types=[ConfigPrimitive(bool)],
                        parser_args={'action': 'store_true'},
                        description="Enables exporting metrics to a file"))
        self._add_config(
            ConfigField(
                'export_path',
                flags=['--export-path', '-e'],
                default_value='.',
                field_types=[ConfigPrimitive(str)],
                description=
                "Full path to directory in which to store the results"))
        self._add_config(
            ConfigField(
                'filename_model_inference',
                flags=['--filename-model-inference'],
                default_value='metrics-model-inference.csv',
                field_types=[ConfigPrimitive(str)],
                description=
                'Specifies filename for storing model inference metrics'))
        self._add_config(
            ConfigField(
                'filename_model_gpu',
                flags=['--filename-model-gpu'],
                field_types=[ConfigPrimitive(str)],
                default_value='metrics-model-gpu.csv',
                description='Specifies filename for storing model GPU metrics')
        )
        self._add_config(
            ConfigField(
                'filename_server_only',
                flags=['--filename-server-only'],
                field_types=[ConfigPrimitive(str)],
                default_value='metrics-server-only.csv',
                description='Specifies filename for server-only metrics'))
        self._add_config(
            ConfigField(
                'max_retries',
                flags=['-r', '--max-retries'],
                field_types=[ConfigPrimitive(int)],
                default_value=100,
                description=
                'Specifies the max number of retries for any retry attempt'))
        self._add_config(
            ConfigField(
                'duration_seconds',
                field_types=[ConfigPrimitive(int)],
                flags=['-d', '--duration-seconds'],
                default_value=5,
                description=
                'Specifies how long (seconds) to gather server-only metrics'))
        self._add_config(
            ConfigField(
                'monitoring_interval',
                flags=['-i', '--monitoring-interval'],
                field_types=[ConfigPrimitive(float)],
                default_value=0.01,
                description=
                'Interval of time between DGCM measurements in seconds'))
        self._add_config(
            ConfigField(
                'client_protocol',
                flags=['--client-protocol'],
                choices=['http', 'grpc'],
                field_types=[ConfigPrimitive(str)],
                default_value='grpc',
                description=
                'The protocol used to communicate with the Triton Inference Server'
            ))
        self._add_config(
            ConfigField(
                'perf_analyzer_path',
                flags=['--perf-analyzer-path'],
                field_types=[ConfigPrimitive(str)],
                default_value='perf_analyzer',
                description=
                'The full path to the perf_analyzer binary executable'))
        self._add_config(
            ConfigField(
                'perf_measurement_window',
                flags=['--perf-measurement-window'],
                field_types=[ConfigPrimitive(int)],
                default_value=5000,
                description=
                'Time interval in milliseconds between perf_analyzer measurements. perf_analyzer will take '
                'measurements over all the requests completed within this time interval.'
            ))
        self._add_config(
            ConfigField(
                'no_perf_output',
                flags=['--no-perf-output'],
                field_types=[ConfigPrimitive(bool)],
                parser_args={'action': 'store_true'},
                description=
                'Silences the output from the perf_analyzer to stdout'))
        self._add_config(
            ConfigField(
                'triton_launch_mode',
                field_types=[ConfigPrimitive(str)],
                flags=['--triton-launch-mode'],
                default_value='local',
                choices=['local', 'docker', 'remote'],
                description="The method by which to launch Triton Server. "
                "'local' assumes tritonserver binary is available locally. "
                "'docker' pulls and launches a triton docker container with "
                "the specified version. 'remote' connects to a running "
                "server using given http, grpc and metrics endpoints. "))
        self._add_config(
            ConfigField('triton_version',
                        flags=['--triton-version'],
                        field_types=[ConfigPrimitive(str)],
                        default_value='20.11-py3',
                        description='Triton Server Docker version'))
        self._add_config(
            ConfigField('log_level',
                        flags=['--log-level'],
                        default_value='INFO',
                        field_types=[ConfigPrimitive(str)],
                        choices=['INFO', 'DEBUG', 'ERROR', 'WARNING'],
                        description='Logging levels'))
        self._add_config(
            ConfigField(
                'triton_http_endpoint',
                default_value='localhost:8000',
                flags=['--triton-http-endpoint'],
                field_types=[ConfigPrimitive(str)],
                description=
                "Triton Server HTTP endpoint url used by Model Analyzer client. "
                "Will be ignored if server-launch-mode is not 'remote'"))
        self._add_config(
            ConfigField(
                'triton_grpc_endpoint',
                flags=['--triton-grpc-endpoint'],
                field_types=[ConfigPrimitive(str)],
                default_value='localhost:8001',
                description=
                "Triton Server HTTP endpoint url used by Model Analyzer client. "
                "Will be ignored if server-launch-mode is not 'remote'"))
        self._add_config(
            ConfigField(
                'triton_metrics_url',
                field_types=[ConfigPrimitive(str)],
                flags=['--triton-metrics-url'],
                default_value='http://localhost:8002/metrics',
                description="Triton Server Metrics endpoint url. "
                "Will be ignored if server-launch-mode is not 'remote'"))
        self._add_config(
            ConfigField('triton_server_path',
                        field_types=[ConfigPrimitive(str)],
                        flags=['--triton-server-path'],
                        default_value='tritonserver',
                        description=
                        'The full path to the tritonserver binary executable'))
        self._add_config(
            ConfigField(
                'triton_output_path',
                field_types=[ConfigPrimitive(str)],
                flags=['--triton-output-path'],
                description=
                'The full path to a file to write the Triton Server log output to.'
            ))
        self._add_config(
            ConfigField(
                'gpus',
                flags=['--gpus'],
                field_types=[ConfigListString()],
                default_value='all',
                description="List of GPU UUIDs to be used for the profiling. "
                "Use 'all' to profile all the GPUs visible by CUDA."))
        self._add_config(
            ConfigField('config_file',
                        field_types=[ConfigPrimitive(str)],
                        flags=['-f', '--config-file'],
                        description="Path to Model Analyzer Config File."))

    def _preprocess_and_verify_arguments(self):
        """
        Enforces some rules on the config.

        Raises
        ------
        TritonModelAnalyzerException
            If there is a problem with arguments or config.
        """

        if self.export:
            if not self.export_path:
                logger.warning(
                    "--export-path specified without --export flag: skipping exporting metrics."
                )
                self.export_path = None
            elif self.export_path and not os.path.isdir(self.export_path):
                raise TritonModelAnalyzerException(
                    f"Export path {self.export_path} is not a directory.")
        if self.triton_launch_mode == 'remote':
            if self.client_protocol == 'http' and not self.triton_http_endpoint:
                raise TritonModelAnalyzerException(
                    "client-protocol is 'http'. Must specify triton-http-endpoint "
                    "if connecting to already running server or change protocol using "
                    "--client-protocol.")
            if self.client_protocol == 'grpc' and not self.triton_grpc_endpoint:
                raise TritonModelAnalyzerException(
                    "client-protocol is 'grpc'. Must specify triton-grpc-endpoint "
                    "if connecting to already running server or change protocol using "
                    "--client-protocol.")

    def _load_config_file(self, file_path):
        """
        Load YAML config

        Parameters
        ----------
        file_path : str
            Path to the Model Analyzer config file
        """

        with open(file_path, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
            return config

    def _setup_logger(self):
        """
        Setup logger format
        """
        log_level = logging.getLevelName(self.log_level)
        logging.basicConfig(level=log_level,
                            format="%(asctime)s.%(msecs)d %(levelname)-4s"
                            "[%(filename)s:%(lineno)d] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    def set_config_values(self, args):
        """
        Set the config values. This function sets all the values for the
        config. CLI arguments have the highest priority, then YAML config
        values and then default values.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed arguments from the CLI

        Raises
        ------
        TritonModelAnalyzerException
            If the required fields are not specified, it will raise
            this exception
        """

        # Config file has been specified
        if 'config_file' in args:
            yaml_config = self._load_config_file(args.config_file)
        else:
            yaml_config = None
        for key, value in self._fields.items():
            if key in args:
                self._fields[key].set_value(getattr(args, key))
            elif yaml_config is not None and key in yaml_config:
                self._fields[key].set_value(yaml_config[key])
            elif value.default_value() is not None:
                self._fields[key].set_value(value.default_value())
            elif value.required():
                flags = ', '.join(value.flags())
                raise TritonModelAnalyzerException(
                    f'Config for {value.name()} is not specified. You need to specify it using the YAML config file or using the {flags} flags in CLI.'
                )
        self._setup_logger()
        self._preprocess_and_verify_arguments()

    def get_config(self):
        """
        Get the config dictionary.

        Returns
        -------
        dict
            Returns a dictionary where the keys are the
            configuration name and the values are ConfigField objects.
        """

        return self._fields

    def get_all_config(self):
        """
        Get a dictionary containing all the configurations.

        Returns
        -------
        dict
            A dictionary containing all the configurations.
        """

        config_dict = {}
        for config in self._fields.values():
            config_dict[config.name()] = config.value()

        return config_dict

    def __getattr__(self, name):
        return self._fields[name].value()
