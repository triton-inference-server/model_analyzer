# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.config.input.config_utils \
    import binary_path_validator, objective_list_output_mapper, file_path_validator, parent_path_validator
from .config_field import ConfigField
from .config_primitive import ConfigPrimitive
from .config_list_string import ConfigListString
from .config_list_numeric import ConfigListNumeric
from .config_object import ConfigObject
from .config_list_generic import ConfigListGeneric
from .config_union import ConfigUnion
from .config_none import ConfigNone
from .config_sweep import ConfigSweep
from .config_enum import ConfigEnum
from .config_command import ConfigCommand

from .config_defaults import \
    DEFAULT_BATCH_SIZES, DEFAULT_CHECKPOINT_DIRECTORY, \
    DEFAULT_CLIENT_PROTOCOL, DEFAULT_DURATION_SECONDS, \
    DEFAULT_GPUS, DEFAULT_MAX_RETRIES, \
    DEFAULT_MONITORING_INTERVAL, DEFAULT_COLLECT_CPU_METRICS, DEFAULT_OFFLINE_OBJECTIVES, \
    DEFAULT_OUTPUT_MODEL_REPOSITORY, DEFAULT_OVERRIDE_OUTPUT_REPOSITORY_FLAG, \
    DEFAULT_PERF_ANALYZER_CPU_UTIL, DEFAULT_PERF_ANALYZER_PATH, DEFAULT_PERF_MAX_AUTO_ADJUSTS, \
    DEFAULT_PERF_OUTPUT_FLAG, DEFAULT_RUN_CONFIG_MAX_CONCURRENCY, \
    DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT, DEFAULT_RUN_CONFIG_PROFILE_MODELS_CONCURRENTLY_ENABLE, \
    DEFAULT_RUN_CONFIG_SEARCH_DISABLE, DEFAULT_TRITON_DOCKER_IMAGE, DEFAULT_TRITON_GRPC_ENDPOINT, \
    DEFAULT_TRITON_HTTP_ENDPOINT, DEFAULT_TRITON_INSTALL_PATH, DEFAULT_TRITON_LAUNCH_MODE, DEFAULT_TRITON_METRICS_URL, \
    DEFAULT_TRITON_SERVER_PATH, DEFAULT_PERF_ANALYZER_TIMEOUT, DEFAULT_USE_LOCAL_GPU_MONITOR

from model_analyzer.constants import LOGGER_NAME
from model_analyzer.triton.server.server_config import \
    TritonServerConfig
from model_analyzer.perf_analyzer.perf_config import \
    PerfAnalyzerConfig
from model_analyzer.record.record import RecordType
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from .objects.config_model_profile_spec import ConfigModelProfileSpec
from .objects.config_protobuf_utils import \
    is_protobuf_type_primitive, protobuf_to_config_type

from tritonclient.grpc.model_config_pb2 import ModelConfig
from google.protobuf.descriptor import FieldDescriptor

import numba
from numba import cuda
import psutil
import logging

logger = logging.getLogger(LOGGER_NAME)


class ConfigCommandProfile(ConfigCommand):
    """
    Model Analyzer config object.
    """

    def _resolve_protobuf_field(self, field):
        """
        Recursively resolve protobuf fields.

        Parameters
        ----------
        field : google.protobuf.pyext._message.FieldDescriptor

        Returns
        -------
        ConfigValue
            A config type equivalent to the protobuf type.

        Raises
        ------
        TritonModelAnalyzerException
            If the protobuf config field cannot be resolved, this exception
            will be raised.
        """

        if is_protobuf_type_primitive(field.type):
            config_type = protobuf_to_config_type(field.type)

            # If it is a repeated field, we should use ConfigListGeneric
            if field.label == FieldDescriptor.LABEL_REPEATED:
                config_type = ConfigListGeneric(ConfigPrimitive(config_type))
            else:
                config_type = ConfigPrimitive(config_type)

        elif field.type == FieldDescriptor.TYPE_MESSAGE:
            # If the field type is TYPE_MESSAGE, we need to create a new
            # message of type ConfigObject
            sub_field_schema = {}

            # Custom handling for map field
            # TODO: Add support for types in the keys
            if field.message_type.has_options and field.message_type.GetOptions(
            ).map_entry:
                value_field_type = self._resolve_protobuf_field(
                    field.message_type.fields_by_name['value'])
                sub_field_schema['*'] = value_field_type
                config_type = ConfigObject(schema=sub_field_schema)

            else:
                fields = field.message_type.fields
                for sub_field in fields:
                    sub_field_schema[
                        sub_field.name] = self._resolve_protobuf_field(
                            sub_field)
                if field.label == FieldDescriptor.LABEL_REPEATED:
                    config_type = ConfigListGeneric(
                        ConfigObject(schema=sub_field_schema))
                else:
                    config_type = ConfigObject(schema=sub_field_schema)
        elif field.type == FieldDescriptor.TYPE_ENUM:
            choices = []
            enum_values = field.enum_type.values
            for enum_value in enum_values:
                choices.append(enum_value.name)
            config_type = ConfigEnum(choices)
        else:
            raise TritonModelAnalyzerException(
                'The current version of Model Config is not supported by Model Analyzer.'
            )

        return ConfigSweep(ConfigUnion([config_type, ConfigNone()]))

    def _get_model_config_fields(self):
        """
        Constructs a ConfigObject from the ModelConfig protobuf.
        """

        model_config_prototype = ModelConfig()
        fields = model_config_prototype.DESCRIPTOR.fields

        schema = {}
        for field in fields:
            schema[field.name] = self._resolve_protobuf_field(field)

        return ConfigObject(schema)

    def _fill_config(self):
        """
        Builder function makes calls to add config to 
        fill the config with options
        """

        self._add_config(
            ConfigField(
                'config_file',
                field_type=ConfigPrimitive(str),
                flags=['-f', '--config-file'],
                description="Path to Config File for subcommand 'profile'."))
        self._add_config(
            ConfigField(
                'checkpoint_directory',
                flags=['-s', '--checkpoint-directory'],
                default_value=DEFAULT_CHECKPOINT_DIRECTORY,
                field_type=ConfigPrimitive(str,
                                           validator=parent_path_validator),
                description=
                "Full path to directory to which to read and write checkpoints and profile data."
            ))
        self._add_config(
            ConfigField(
                'monitoring_interval',
                flags=['-i', '--monitoring-interval'],
                field_type=ConfigPrimitive(float),
                default_value=DEFAULT_MONITORING_INTERVAL,
                description=
                'Interval of time between metrics measurements in seconds'))
        self._add_config(
            ConfigField(
                'duration_seconds',
                field_type=ConfigPrimitive(int),
                flags=['-d', '--duration-seconds'],
                default_value=DEFAULT_DURATION_SECONDS,
                description=
                'Specifies how long (seconds) to gather server-only metrics'))
        self._add_config(
            ConfigField(
                'use_local_gpu_monitor',
                field_type=ConfigPrimitive(bool),
                flags=['--use-local-gpu-monitor'],
                parser_args={'action': 'store_true'},
                default_value=DEFAULT_USE_LOCAL_GPU_MONITOR,
                description=
                'Specify whether GPU metrics should be monitored by local DCGM monitor. '
                'If this flag is set, model analyzer will query metrics directly via DCGM.'
            ))
        self._add_config(
            ConfigField(
                'collect_cpu_metrics',
                field_type=ConfigPrimitive(bool),
                flags=['--collect-cpu-metrics'],
                parser_args={'action': 'store_true'},
                default_value=DEFAULT_COLLECT_CPU_METRICS,
                description='Specify whether CPU metrics are collected or not'))
        self._add_config(
            ConfigField(
                'gpus',
                flags=['--gpus'],
                field_type=ConfigListString(),
                default_value=DEFAULT_GPUS,
                description="List of GPU UUIDs to be used for the profiling. "
                "Use 'all' to profile all the GPUs visible by CUDA."))

        self._add_repository_configs()
        self._add_client_configs()
        self._add_profile_models_configs()
        self._add_perf_analyzer_configs()
        self._add_triton_configs()
        self._add_run_search_configs()

    def _add_repository_configs(self):
        """
        Adds configs specific to model repository
        """
        self._add_config(
            ConfigField('model_repository',
                        flags=['-m', '--model-repository'],
                        field_type=ConfigPrimitive(
                            str, required=True, validator=file_path_validator),
                        description='Model repository location'))
        self._add_config(
            ConfigField(
                'output_model_repository_path',
                field_type=ConfigPrimitive(str),
                default_value=DEFAULT_OUTPUT_MODEL_REPOSITORY,
                flags=['--output-model-repository-path'],
                description='Output model repository path used by Model Analyzer.'
                ' This is the directory that will contain all the generated model configurations'
            ))
        self._add_config(
            ConfigField(
                'override_output_model_repository',
                field_type=ConfigPrimitive(bool),
                parser_args={'action': 'store_true'},
                default_value=DEFAULT_OVERRIDE_OUTPUT_REPOSITORY_FLAG,
                flags=['--override-output-model-repository'],
                description=
                'Will override the contents of the output model repository'
                ' and replace it with the new results.'))

    def _add_profile_models_configs(self):
        """
        Adds configs specific to model specifications
        """
        triton_server_flags_scheme = ConfigObject(schema={
            k: ConfigPrimitive(str) for k in TritonServerConfig.allowed_keys()
        })
        perf_analyzer_additive_keys = {
            k: None for k in PerfAnalyzerConfig.additive_keys()
        }
        perf_analyzer_flags_scheme = ConfigObject(
            schema={
                k:
                ((ConfigUnion([ConfigPrimitive(
                    type_=str), ConfigListString()])) if (
                        k in perf_analyzer_additive_keys) else ConfigPrimitive(
                            type_=str))
                for k in PerfAnalyzerConfig.allowed_keys()
            })
        triton_server_environment_scheme = ConfigObject(
            schema={'*': ConfigPrimitive(str)})
        self._add_config(
            ConfigField(
                'perf_analyzer_flags',
                field_type=perf_analyzer_flags_scheme,
                description=
                'Allows custom configuration of the perf analyzer instances used by model analyzer.'
            ))
        self._add_config(
            ConfigField(
                'triton_server_flags',
                field_type=triton_server_flags_scheme,
                description=
                'Allows custom configuration of the triton instances used by model analyzer.'
            ))
        self._add_config(
            ConfigField(
                'triton_server_environment',
                field_type=triton_server_environment_scheme,
                description=
                'Allows setting environment variables for tritonserver server instances launched by Model Analyzer'
            ))

        objectives_scheme = ConfigUnion([
            ConfigObject(
                schema={
                    tag: ConfigPrimitive(type_=int)
                    for tag in RecordType.get_all_record_types().keys()
                }),
            ConfigListString(output_mapper=objective_list_output_mapper)
        ])
        constraints_scheme = ConfigObject(
            schema={
                'perf_throughput':
                    ConfigObject(schema={
                        'min': ConfigPrimitive(int),
                    }),
                'perf_latency_avg':
                    ConfigObject(schema={
                        'max': ConfigPrimitive(int),
                    }),
                'perf_latency_p90':
                    ConfigObject(schema={
                        'max': ConfigPrimitive(int),
                    }),
                'perf_latency_p95':
                    ConfigObject(schema={
                        'max': ConfigPrimitive(int),
                    }),
                'perf_latency_p99':
                    ConfigObject(schema={
                        'max': ConfigPrimitive(int),
                    }),
                'perf_latency':
                    ConfigObject(schema={
                        'max': ConfigPrimitive(int),
                    }),
                'gpu_used_memory':
                    ConfigObject(schema={
                        'max': ConfigPrimitive(int),
                    }),
            })
        self._add_config(
            ConfigField(
                'objectives',
                field_type=objectives_scheme,
                default_value=DEFAULT_OFFLINE_OBJECTIVES,
                description=
                'Model Analyzer uses the objectives described here to find the best configuration for each model.'
            ))
        self._add_config(
            ConfigField(
                'constraints',
                field_type=constraints_scheme,
                description=
                'Constraints on the objectives specified in the "objectives" field of the config.'
            ))
        model_config_fields = self._get_model_config_fields()
        profile_model_scheme = ConfigObject(
            required=True,
            schema={
                # Any key is allowed, but the keys must follow the pattern
                # below
                '*':
                    ConfigObject(
                        schema={
                            'cpu_only':
                                ConfigPrimitive(bool),
                            'parameters':
                                ConfigObject(
                                    schema={
                                        'batch_sizes':
                                            ConfigListNumeric(type_=int),
                                        'concurrency':
                                            ConfigListNumeric(type_=int)
                                    }),
                            'objectives':
                                objectives_scheme,
                            'constraints':
                                constraints_scheme,
                            'model_config_parameters':
                                model_config_fields,
                            'perf_analyzer_flags':
                                perf_analyzer_flags_scheme,
                            'triton_server_flags':
                                triton_server_flags_scheme,
                            'triton_server_environment':
                                triton_server_environment_scheme
                        })
            },
            output_mapper=ConfigModelProfileSpec.
            model_object_to_config_model_profile_spec)
        self._add_config(
            ConfigField(
                'profile_models',
                flags=['--profile-models'],
                field_type=ConfigUnion([
                    profile_model_scheme,
                    ConfigListGeneric(ConfigUnion([
                        profile_model_scheme,
                        ConfigPrimitive(str,
                                        output_mapper=ConfigModelProfileSpec.
                                        model_str_to_config_model_profile_spec)
                    ]),
                                      required=True,
                                      output_mapper=ConfigModelProfileSpec.
                                      model_mixed_to_config_model_profile_spec),
                    ConfigListString(output_mapper=ConfigModelProfileSpec.
                                     model_list_to_config_model_profile_spec),
                ],
                                       required=True),
                description='List of the models to be profiled'))
        self._add_config(
            ConfigField(
                'batch_sizes',
                flags=['-b', '--batch-sizes'],
                field_type=ConfigListNumeric(int),
                default_value=DEFAULT_BATCH_SIZES,
                description=
                'Comma-delimited list of batch sizes to use for the profiling'))
        self._add_config(
            ConfigField(
                'concurrency',
                flags=['-c', '--concurrency'],
                field_type=ConfigListNumeric(int),
                description=
                "Comma-delimited list of concurrency values or ranges <start:end:step>"
                " to be used during profiling"))
        self._add_config(
            ConfigField(
                'reload_model_disable',
                field_type=ConfigPrimitive(bool),
                parser_args={'action': 'store_true'},
                default_value=False,
                flags=['--reload-model-disable'],
                description='Flag to indicate whether or not to disable model '
                'loading and unloading in remote mode.'))

    def _add_client_configs(self):
        """
        Adds configs specific to tritonclient
        """
        self._add_config(
            ConfigField(
                'client_max_retries',
                flags=['-r', '--client-max-retries'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_MAX_RETRIES,
                description=
                'Specifies the max number of retries for any requests to Triton server.'
            ))
        self._add_config(
            ConfigField(
                'client_protocol',
                flags=['--client-protocol'],
                choices=['http', 'grpc'],
                field_type=ConfigPrimitive(str),
                default_value=DEFAULT_CLIENT_PROTOCOL,
                description=
                'The protocol used to communicate with the Triton Inference Server'
            ))

    def _add_run_search_configs(self):
        """
        Add the config options related
        to the run search
        """

        self._add_config(
            ConfigField(
                'run_config_search_max_concurrency',
                flags=['--run-config-search-max-concurrency'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_RUN_CONFIG_MAX_CONCURRENCY,
                description=
                "Max concurrency value that run config search should not go beyond that."
            ))
        self._add_config(
            ConfigField(
                'run_config_search_max_instance_count',
                flags=['--run-config-search-max-instance-count'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT,
                description=
                "Max instance count value that run config search should not go beyond that."
            ))
        self._add_config(
            ConfigField('run_config_search_disable',
                        flags=['--run-config-search-disable'],
                        field_type=ConfigPrimitive(bool),
                        parser_args={'action': 'store_true'},
                        default_value=DEFAULT_RUN_CONFIG_SEARCH_DISABLE,
                        description="Disable run config search."))
        # FIXME: TODO-MM:  Disable until we support this feature
        # self._add_config(
        #     ConfigField(
        #         'run_config_profile_models_concurrently_enable',
        #         flags=['--run-config-profile-models-concurrently-enable'],
        #         field_type=ConfigPrimitive(bool),
        #         parser_args={'action': 'store_true'},
        #         default_value=
        #         DEFAULT_RUN_CONFIG_PROFILE_MODELS_CONCURRENTLY_ENABLE,
        #         description=
        #         "Enable the profiling of all supplied models concurrently."))

    def _add_triton_configs(self):
        """
        Adds the triton related flags
        and config options
        """

        self._add_config(
            ConfigField(
                'triton_launch_mode',
                field_type=ConfigPrimitive(str),
                flags=['--triton-launch-mode'],
                default_value=DEFAULT_TRITON_LAUNCH_MODE,
                choices=['local', 'docker', 'remote', 'c_api'],
                description="The method by which to launch Triton Server. "
                "'local' assumes tritonserver binary is available locally. "
                "'docker' pulls and launches a triton docker container with "
                "the specified version. 'remote' connects to a running "
                "server using given http, grpc and metrics endpoints. "
                "'c_api' allows direct benchmarking of Triton locally"
                "without the use of endpoints."))
        self._add_config(
            ConfigField('triton_docker_image',
                        flags=['--triton-docker-image'],
                        field_type=ConfigPrimitive(str),
                        default_value=DEFAULT_TRITON_DOCKER_IMAGE,
                        description='Triton Server Docker image tag'))
        self._add_config(
            ConfigField(
                'triton_http_endpoint',
                flags=['--triton-http-endpoint'],
                field_type=ConfigPrimitive(str),
                default_value=DEFAULT_TRITON_HTTP_ENDPOINT,
                description=
                "Triton Server HTTP endpoint url used by Model Analyzer client. "
                "Will be ignored if server-launch-mode is not 'remote'"))
        self._add_config(
            ConfigField(
                'triton_grpc_endpoint',
                flags=['--triton-grpc-endpoint'],
                field_type=ConfigPrimitive(str),
                default_value=DEFAULT_TRITON_GRPC_ENDPOINT,
                description=
                "Triton Server HTTP endpoint url used by Model Analyzer client. "
                "Will be ignored if server-launch-mode is not 'remote'"))
        self._add_config(
            ConfigField(
                'triton_metrics_url',
                field_type=ConfigPrimitive(str),
                flags=['--triton-metrics-url'],
                default_value=DEFAULT_TRITON_METRICS_URL,
                description="Triton Server Metrics endpoint url. "
                "Will be ignored if server-launch-mode is not 'remote'"))
        self._add_config(
            ConfigField(
                'triton_server_path',
                field_type=ConfigPrimitive(str),
                flags=['--triton-server-path'],
                default_value=DEFAULT_TRITON_SERVER_PATH,
                description='The full path to the tritonserver binary executable'
            ))
        self._add_config(
            ConfigField(
                'triton_output_path',
                field_type=ConfigPrimitive(str),
                flags=['--triton-output-path'],
                description=
                ('The full path to the file to which Triton server instance will '
                 'append their log output. If not specified, they are not written.'
                )))
        self._add_config(
            ConfigField(
                'triton_docker_mounts',
                field_type=ConfigListString(),
                flags=['--triton-docker-mounts'],
                description=
                ("A list of strings representing volumes to be mounted. "
                 "The strings should have the format '<host path>:<container path>:<access mode>'."
                )))
        self._add_config(
            ConfigField(
                'triton_docker_labels',
                field_type=ConfigObject(schema={'*': ConfigPrimitive(str)}),
                description=
                'A dictionary of name-value labels to set metadata for the Triton '
                'server docker container in docker launch mode'))
        self._add_config(
            ConfigField(
                'triton_docker_shm_size',
                field_type=ConfigPrimitive(str),
                flags=['--triton-docker-shm-size'],
                description=(
                    'The size of the /dev/shm for the triton docker container'
                )))
        self._add_config(
            ConfigField(
                'triton_install_path',
                field_type=ConfigPrimitive(str),
                default_value=DEFAULT_TRITON_INSTALL_PATH,
                flags=['--triton-install-path'],
                description=
                ("Path to Triton install directory i.e. the parent directory of 'lib/libtritonserver.so'."
                 "Required only when using triton_launch_mode=c_api.")))

    def _add_perf_analyzer_configs(self):
        """
        Add the perf_analyzer related config
        options
        """

        self._add_config(
            ConfigField('perf_analyzer_timeout',
                        flags=['--perf-analyzer-timeout'],
                        field_type=ConfigPrimitive(int),
                        default_value=DEFAULT_PERF_ANALYZER_TIMEOUT,
                        description="Perf analyzer timeout value in seconds."))
        self._add_config(
            ConfigField(
                'perf_analyzer_cpu_util',
                flags=['--perf-analyzer-cpu-util'],
                field_type=ConfigPrimitive(float),
                default_value=psutil.cpu_count() *
                DEFAULT_PERF_ANALYZER_CPU_UTIL,
                description=
                "Maximum CPU utilization value allowed for the perf_analyzer."))
        self._add_config(
            ConfigField('perf_analyzer_path',
                        flags=['--perf-analyzer-path'],
                        field_type=ConfigPrimitive(
                            str, validator=binary_path_validator),
                        default_value=DEFAULT_PERF_ANALYZER_PATH,
                        description=
                        'The full path to the perf_analyzer binary executable'))
        self._add_config(
            ConfigField(
                'perf_output',
                flags=['--perf-output'],
                parser_args={'action': 'store_true'},
                field_type=ConfigPrimitive(bool),
                default_value=DEFAULT_PERF_OUTPUT_FLAG,
                description=
                'Enables the output from the perf_analyzer to a file specified by'
                ' perf_output_path. If perf_output_path is None, output will be'
                ' written to stdout.'))
        self._add_config(
            ConfigField(
                'perf_output_path',
                flags=['--perf-output-path'],
                field_type=ConfigPrimitive(str),
                description=
                'Path to the file to which write perf_analyzer output, if enabled.'
            ))
        self._add_config(
            ConfigField(
                'perf_analyzer_max_auto_adjusts',
                flags=['--perf-analyzer-max-auto-adjusts'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_PERF_MAX_AUTO_ADJUSTS,
                description="Maximum number of times perf_analyzer is "
                "launched with auto adjusted parameters in an attempt to profile a model. "
            ))

    def _preprocess_and_verify_arguments(self):
        """
        Enforces some rules on the config.

        Raises
        ------
        TritonModelAnalyzerException
            If there is a problem with arguments or config.
        """

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
        elif self.triton_docker_mounts or self.triton_docker_labels:
            if self.triton_launch_mode == 'docker':
                # Verify format
                if self.triton_docker_mounts:
                    for volume_str in self.triton_docker_mounts:
                        if volume_str.count(':') != 2:
                            raise TritonModelAnalyzerException(
                                "triton_docker_mounts needs to be a list of strings. Each string "
                                " should be of the format <host path>:<container dest>:<access mode>"
                            )
            else:
                logger.warning(
                    f"Triton launch mode is set to {self.triton_launch_mode}. "
                    "Ignoring triton_docker_mounts and triton_docker_labels.")

        if self.triton_launch_mode == 'docker':
            if not self.triton_docker_image or self.triton_docker_image.isspace(
            ):
                raise TritonModelAnalyzerException(
                    "triton_docker_image provided but is empty.")

        if self.triton_launch_mode == 'c_api':
            if self.triton_server_flags:
                logger.warning(
                    "Triton launch mode is set to C_API. Model Analyzer cannot set "
                    "triton_server_flags.")
            if self.triton_output_path:
                logger.warning(
                    "Triton launch mode is set to C_API, triton logs are not supported. "
                    "Triton server error output can be obtained by setting perf_output_path."
                )
        # If run config search is disabled and no concurrency value is provided,
        # set the default value.
        if self.run_config_search_disable:
            if len(self.concurrency) == 0:
                self.concurrency = [1]

    def _autofill_values(self):
        """
        Fill in the implied or default
        config values.
        """

        cpu_only = False
        if len(self.gpus) == 0 or not cuda.is_available():
            cpu_only = True

        new_profile_models = {}
        for model in self.profile_models:
            new_model = {'cpu_only': (model.cpu_only() or cpu_only)}

            # Objectives
            if not model.objectives():
                new_model['objectives'] = self.objectives
            else:
                new_model['objectives'] = model.objectives()

            # Constraints
            if not model.constraints():
                if 'constraints' in self._fields and self._fields[
                        'constraints'].value():
                    new_model['constraints'] = self.constraints
            else:
                new_model['constraints'] = model.constraints()

            # Run parameters
            if not model.parameters():
                new_model['parameters'] = {
                    'batch_sizes': self.batch_sizes,
                    'concurrency': self.concurrency
                }
            elif 'batch_sizes' not in model.parameters():
                new_model['parameters'] = {
                    'batch_sizes': self.batch_sizes,
                    'concurrency': model.parameters()['concurrency']
                }
            elif 'concurrency' not in model.parameters():
                new_model['parameters'] = {
                    'batch_sizes': model.parameters()['batch_sizes'],
                    'concurrency': self.concurrency
                }
            else:
                new_model['parameters'] = model.parameters()

            # Perf analyzer flags
            if not model.perf_analyzer_flags():
                new_model['perf_analyzer_flags'] = self.perf_analyzer_flags
            else:
                new_model['perf_analyzer_flags'] = model.perf_analyzer_flags()

            # triton server flags
            if not model.triton_server_flags():
                new_model['triton_server_flags'] = self.triton_server_flags
            else:
                new_model['triton_server_flags'] = model.triton_server_flags()

            # triton server env
            if not model.triton_server_environment():
                new_model[
                    'triton_server_environment'] = self.triton_server_environment
            else:
                new_model[
                    'triton_server_environment'] = model.triton_server_environment(
                    )

            # Transfer model config parameters directly
            if model.model_config_parameters():
                new_model[
                    'model_config_parameters'] = model.model_config_parameters(
                    )

            new_profile_models[model.model_name()] = new_model
        self._fields['profile_models'].set_value(new_profile_models)
