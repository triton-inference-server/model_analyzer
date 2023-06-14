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
    DEFAULT_GPUS, DEFAULT_SKIP_SUMMARY_REPORTS, DEFAULT_MAX_RETRIES, \
    DEFAULT_MONITORING_INTERVAL, DEFAULT_COLLECT_CPU_METRICS, DEFAULT_OFFLINE_OBJECTIVES, \
    DEFAULT_OUTPUT_MODEL_REPOSITORY, DEFAULT_OVERRIDE_OUTPUT_REPOSITORY_FLAG, \
    DEFAULT_PERF_ANALYZER_CPU_UTIL, DEFAULT_PERF_ANALYZER_PATH, DEFAULT_PERF_MAX_AUTO_ADJUSTS, \
    DEFAULT_PERF_OUTPUT_FLAG, DEFAULT_RUN_CONFIG_MAX_CONCURRENCY, DEFAULT_RUN_CONFIG_MIN_CONCURRENCY, \
    DEFAULT_RUN_CONFIG_MAX_REQUEST_RATE, DEFAULT_RUN_CONFIG_MIN_REQUEST_RATE, \
    DEFAULT_RUN_CONFIG_PROFILE_MODELS_CONCURRENTLY_ENABLE, DEFAULT_RUN_CONFIG_SEARCH_MODE, \
    DEFAULT_RUN_CONFIG_MAX_BINARY_SEARCH_STEPS, DEFAULT_REQUEST_RATE_SEARCH_ENABLE, \
    DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT, DEFAULT_RUN_CONFIG_MIN_INSTANCE_COUNT, \
    DEFAULT_RUN_CONFIG_MAX_MODEL_BATCH_SIZE, DEFAULT_RUN_CONFIG_MIN_MODEL_BATCH_SIZE, \
    DEFAULT_RUN_CONFIG_SEARCH_DISABLE, DEFAULT_TRITON_DOCKER_IMAGE, DEFAULT_TRITON_GRPC_ENDPOINT, \
    DEFAULT_TRITON_HTTP_ENDPOINT, DEFAULT_TRITON_INSTALL_PATH, DEFAULT_TRITON_LAUNCH_MODE, DEFAULT_TRITON_METRICS_URL, \
    DEFAULT_TRITON_SERVER_PATH, DEFAULT_PERF_ANALYZER_TIMEOUT, \
    DEFAULT_EXPORT_PATH, DEFAULT_FILENAME_MODEL_INFERENCE, DEFAULT_FILENAME_MODEL_GPU, \
    DEFAULT_FILENAME_SERVER_ONLY, DEFAULT_NUM_CONFIGS_PER_MODEL, DEFAULT_NUM_TOP_MODEL_CONFIGS, \
    DEFAULT_INFERENCE_OUTPUT_FIELDS, DEFAULT_REQUEST_RATE_INFERENCE_OUTPUT_FIELDS, \
    DEFAULT_GPU_OUTPUT_FIELDS, DEFAULT_REQUEST_RATE_GPU_OUTPUT_FIELDS, DEFAULT_SERVER_OUTPUT_FIELDS, \
    DEFAULT_ONLINE_OBJECTIVES, DEFAULT_ONLINE_PLOTS, DEFAULT_OFFLINE_PLOTS, DEFAULT_MODEL_WEIGHTING, \
    DEFAULT_SKIP_DETAILED_REPORTS

from model_analyzer.constants import LOGGER_NAME
from model_analyzer.triton.server.server_config import \
    TritonServerConfig
from model_analyzer.perf_analyzer.perf_config import \
    PerfAnalyzerConfig
from model_analyzer.record.record import RecordType
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from .objects.config_plot import ConfigPlot
from .objects.config_model_profile_spec import ConfigModelProfileSpec
from .objects.config_protobuf_utils import \
    is_protobuf_type_primitive, protobuf_to_config_type

from tritonclient.grpc.model_config_pb2 import ModelConfig
from google.protobuf.descriptor import FieldDescriptor

import os
import numba
import argparse
from numba import cuda
import psutil
import logging

logger = logging.getLogger(LOGGER_NAME)


class ConfigCommandProfile(ConfigCommand):
    """
    Model Analyzer config object.
    """

    def _resolve_protobuf_field(self, field: FieldDescriptor) -> ConfigSweep:
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
        self._add_config(
            ConfigField(
                'skip_summary_reports',
                flags=['--skip-summary-reports'],
                field_type=ConfigPrimitive(bool),
                parser_args={'action': 'store_true'},
                default_value=DEFAULT_SKIP_SUMMARY_REPORTS,
                description=
                'Skips the generation of analysis summary reports and tables.'))
        self._add_config(
            ConfigField(
                'skip_detailed_reports',
                flags=['--skip-detailed-reports'],
                field_type=ConfigPrimitive(bool),
                parser_args={'action': 'store_true'},
                default_value=DEFAULT_SKIP_DETAILED_REPORTS,
                description=
                "Skips the generation of detailed summary reports and tables."))

        self._add_repository_configs()
        self._add_client_configs()
        self._add_profile_models_configs()
        self._add_perf_analyzer_configs()
        self._add_triton_configs()
        self._add_run_search_configs()
        self._add_export_configs()
        self._add_report_configs()
        self._add_table_configs()
        self._add_shorthand_configs()

    def _add_repository_configs(self):
        """
        Adds configs specific to model repository
        """
        self._add_config(
            ConfigField('model_repository',
                        flags=['-m', '--model-repository'],
                        field_type=ConfigPrimitive(
                            str, validator=file_path_validator),
                        description='Triton Model repository location'))
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

        # This comes from the installed python package:
        # <install_path>/lib/python3.8/dist-packages/docker/models/containers.py
        # Only supporting values that are bool, int, string, or lists of strings
        triton_docker_args_scheme = ConfigObject(
            schema={
                'image': ConfigPrimitive(str),
                'command': ConfigPrimitive(str),
                'auto_remove': ConfigPrimitive(bool),
                'blkio_weight_device': ConfigListString(),
                'blkio_weight': ConfigPrimitive(int),
                'cap_add': ConfigListString(),
                'cap_drop': ConfigListString(),
                'cgroup_parent': ConfigPrimitive(str),
                'cgroupns': ConfigPrimitive(str),
                'cpu_count': ConfigPrimitive(int),
                'cpu_percent': ConfigPrimitive(int),
                'cpu_period': ConfigPrimitive(int),
                'cpu_quota': ConfigPrimitive(int),
                'cpu_rt_period': ConfigPrimitive(int),
                'cpu_shares': ConfigPrimitive(int),
                'cpuset_cpus': ConfigPrimitive(str),
                'cpuset_mems': ConfigPrimitive(str),
                'detach': ConfigPrimitive(bool),
                'domainname': ConfigPrimitive(str),
                'entrypoint': ConfigPrimitive(str),
                'environment': ConfigListString(),
                'hostname': ConfigPrimitive(str),
                'init': ConfigPrimitive(bool),
                'init_path': ConfigPrimitive(str),
                'ipc_mode': ConfigPrimitive(str),
                'isolation': ConfigPrimitive(str),
                'kernel_memory': ConfigPrimitive(str),
                'labels': ConfigListString(),
                'mac_address': ConfigPrimitive(str),
                'mem_limit': ConfigPrimitive(str),
                'mem_reservation': ConfigPrimitive(str),
                'memswap_limit': ConfigPrimitive(str),
                'name': ConfigPrimitive(str),
                'nano_cpus': ConfigPrimitive(int),
                'network': ConfigPrimitive(str),
                'network_disabled': ConfigPrimitive(bool),
                'network_mode': ConfigPrimitive(str),
                'oom_kill_disable': ConfigPrimitive(bool),
                'oom_score_adj': ConfigPrimitive(int),
                'pid_mode': ConfigPrimitive(str),
                'pids_limit': ConfigPrimitive(int),
                'platform': ConfigPrimitive(str),
                'privileged': ConfigPrimitive(bool),
                'publish_all_ports': ConfigPrimitive(bool),
                'remove': ConfigPrimitive(bool),
                'runtime': ConfigPrimitive(str),
                'shm_size': ConfigPrimitive(str),
                'stdin_open': ConfigPrimitive(bool),
                'stdout': ConfigPrimitive(bool),
                'stderr': ConfigPrimitive(bool),
                'stop_signal': ConfigPrimitive(str),
                'stream': ConfigPrimitive(bool),
                'tty': ConfigPrimitive(bool),
                'use_config_proxy': ConfigPrimitive(bool),
                'user': ConfigPrimitive(str),
                'userns_mode': ConfigPrimitive(str),
                'uts_mode': ConfigPrimitive(str),
                'version': ConfigPrimitive(str),
                'volume_driver': ConfigPrimitive(str),
                'volumes': ConfigListString(),
                'working_dir': ConfigPrimitive(str)
            })

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
        self._add_config(
            ConfigField(
                'triton_docker_args',
                field_type=triton_docker_args_scheme,
                description=
                'Allows setting docker variables for tritonserver server instances launched by Model Analyzer'
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
        self._add_config(
            ConfigField(
                'weighting',
                field_type=ConfigPrimitive(int),
                description=
                'A weighting used to bias the model when determining the best configuration'
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
                                            ConfigListNumeric(type_=int),
                                        'request_rate':
                                            ConfigListNumeric(type_=int)
                                    }),
                            'objectives':
                                objectives_scheme,
                            'constraints':
                                constraints_scheme,
                            'weighting':
                                ConfigPrimitive(type_=int),
                            'model_config_parameters':
                                model_config_fields,
                            'perf_analyzer_flags':
                                perf_analyzer_flags_scheme,
                            'triton_server_flags':
                                triton_server_flags_scheme,
                            'triton_server_environment':
                                triton_server_environment_scheme,
                            'triton_docker_args':
                                triton_docker_args_scheme
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
                'request_rate',
                flags=['--request-rate'],
                field_type=ConfigListNumeric(int),
                description=
                "Comma-delimited list of request rate values or ranges <start:end:step>"
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
        self._add_config(
            ConfigField(
                'bls_composing_models',
                flags=['--bls-composing-models'],
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
                default_value=[],
                description='List of the models to be profiled'))
        self._add_config(
            ConfigField(
                'cpu_only_composing_models',
                field_type=ConfigListString(),
                flags=['--cpu-only-composing-models'],
                description=
                ("A list of strings representing composing models that should be profiled using CPU instances only. "
                )))

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
                'early_exit_enable',
                field_type=ConfigPrimitive(bool),
                parser_args={'action': 'store_true'},
                default_value=False,
                flags=['--early-exit-enable'],
                description=
                'Flag to indicate if Model Analyzer can skip some configurations when manually searching concurrency/request rate, or max_batch_size'
            ))
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
                'run_config_search_min_concurrency',
                flags=['--run-config-search-min-concurrency'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_RUN_CONFIG_MIN_CONCURRENCY,
                description=
                "Min concurrency value that run config search should start with."
            ))
        self._add_config(
            ConfigField(
                'run_config_search_max_request_rate',
                flags=['--run-config-search-max-request-rate'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_RUN_CONFIG_MAX_REQUEST_RATE,
                description=
                "Max request rate value that run config search should not go beyond that."
            ))
        self._add_config(
            ConfigField(
                'run_config_search_min_request_rate',
                flags=['--run-config-search-min-request-rate'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_RUN_CONFIG_MIN_REQUEST_RATE,
                description=
                "Min request rate value that run config search should start with."
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
            ConfigField(
                'run_config_search_min_instance_count',
                flags=['--run-config-search-min-instance-count'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_RUN_CONFIG_MIN_INSTANCE_COUNT,
                description=
                "Min instance count value that run config search should start with."
            ))
        self._add_config(
            ConfigField(
                'run_config_search_max_model_batch_size',
                flags=['--run-config-search-max-model-batch-size'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_RUN_CONFIG_MAX_MODEL_BATCH_SIZE,
                description=
                "Value for the model's max_batch_size that run config search will not go beyond."
            ))
        self._add_config(
            ConfigField(
                'run_config_search_min_model_batch_size',
                flags=['--run-config-search-min-model-batch-size'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_RUN_CONFIG_MIN_MODEL_BATCH_SIZE,
                description=
                "Value for the model's max_batch_size that run config search will start from."
            ))
        self._add_config(
            ConfigField(
                'run_config_search_max_binary_search_steps',
                flags=['--run-config-search-max-binary-search-steps'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_RUN_CONFIG_MAX_BINARY_SEARCH_STEPS,
                description=
                "Maximum number of steps take during the binary concurrency search."
            ))
        self._add_config(
            ConfigField(
                'run_config_search_mode',
                flags=['--run-config-search-mode'],
                choices=['brute', 'quick'],
                field_type=ConfigPrimitive(str),
                default_value=DEFAULT_RUN_CONFIG_SEARCH_MODE,
                description=
                "The search mode for Model Analyzer to find and evaluate"
                " model configurations. 'brute' will brute force all combinations of"
                " configuration options.  'quick' will attempt to find a near-optimal"
                " configuration as fast as possible, but isn't guaranteed to find the"
                " best."))
        self._add_config(
            ConfigField('run_config_search_disable',
                        flags=['--run-config-search-disable'],
                        field_type=ConfigPrimitive(bool),
                        parser_args={'action': 'store_true'},
                        default_value=DEFAULT_RUN_CONFIG_SEARCH_DISABLE,
                        description="Disable run config search."))
        self._add_config(
            ConfigField(
                'run_config_profile_models_concurrently_enable',
                flags=['--run-config-profile-models-concurrently-enable'],
                field_type=ConfigPrimitive(bool),
                parser_args={'action': 'store_true'},
                default_value=
                DEFAULT_RUN_CONFIG_PROFILE_MODELS_CONCURRENTLY_ENABLE,
                description=
                "Enable the profiling of all supplied models concurrently."))
        self._add_config(
            ConfigField(
                'request_rate_search_enable',
                flags=['--request-rate-search-enable'],
                field_type=ConfigPrimitive(bool),
                parser_args={'action': 'store_true'},
                default_value=DEFAULT_REQUEST_RATE_SEARCH_ENABLE,
                description=
                "Enables the searching of request rate (instead of concurrency)."
            ))

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
                "Triton Server HTTP endpoint url used by Model Analyzer client."
            ))
        self._add_config(
            ConfigField(
                'triton_grpc_endpoint',
                flags=['--triton-grpc-endpoint'],
                field_type=ConfigPrimitive(str),
                default_value=DEFAULT_TRITON_GRPC_ENDPOINT,
                description=
                "Triton Server HTTP endpoint url used by Model Analyzer client."
            ))
        self._add_config(
            ConfigField('triton_metrics_url',
                        field_type=ConfigPrimitive(str),
                        flags=['--triton-metrics-url'],
                        default_value=DEFAULT_TRITON_METRICS_URL,
                        description="Triton Server Metrics endpoint url. "))
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

    def _add_export_configs(self):
        """
        Add configs related to exporting data
        """
        self._add_config(
            ConfigField('export_path',
                        flags=['-e', '--export-path'],
                        default_value=DEFAULT_EXPORT_PATH,
                        field_type=ConfigPrimitive(
                            str, validator=parent_path_validator),
                        description=
                        "Full path to directory in which to store the results"))
        self._add_config(
            ConfigField(
                'filename_model_inference',
                flags=['--filename-model-inference'],
                default_value=DEFAULT_FILENAME_MODEL_INFERENCE,
                field_type=ConfigPrimitive(str),
                description=
                'Specifies filename for storing model inference metrics'))
        self._add_config(
            ConfigField(
                'filename_model_gpu',
                flags=['--filename-model-gpu'],
                field_type=ConfigPrimitive(str),
                default_value=DEFAULT_FILENAME_MODEL_GPU,
                description='Specifies filename for storing model GPU metrics'))
        self._add_config(
            ConfigField(
                'filename_server_only',
                flags=['--filename-server-only'],
                field_type=ConfigPrimitive(str),
                default_value=DEFAULT_FILENAME_SERVER_ONLY,
                description='Specifies filename for server-only metrics'))

    def _add_report_configs(self):
        """
        Adds report related configs
        """
        self._add_config(
            ConfigField(
                'num_configs_per_model',
                flags=['--num-configs-per-model'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_NUM_CONFIGS_PER_MODEL,
                description=
                'The number of configurations to plot per model in the summary.'
            ))
        self._add_config(
            ConfigField(
                'num_top_model_configs',
                flags=['--num-top-model-configs'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_NUM_TOP_MODEL_CONFIGS,
                description=
                'Model Analyzer will compare this many of the top models configs across all models.'
            ))

    def _add_table_configs(self):
        """
        Adds result table related
        configs
        """
        self._add_config(
            ConfigField(
                'inference_output_fields',
                flags=['--inference-output-fields'],
                field_type=ConfigListString(),
                default_value=DEFAULT_INFERENCE_OUTPUT_FIELDS,
                description=
                'Specifies column keys for model inference metrics table'))
        self._add_config(
            ConfigField(
                'gpu_output_fields',
                flags=['--gpu-output-fields'],
                field_type=ConfigListString(),
                default_value=DEFAULT_GPU_OUTPUT_FIELDS,
                description='Specifies column keys for model gpu metrics table')
        )
        self._add_config(
            ConfigField(
                'server_output_fields',
                flags=['--server-output-fields'],
                field_type=ConfigListString(),
                default_value=DEFAULT_SERVER_OUTPUT_FIELDS,
                description='Specifies column keys for server-only metrics table'
            ))

    def _add_shorthand_configs(self):
        """
        Adds configs for various shorthands
        """
        self._add_config(
            ConfigField(
                'latency_budget',
                flags=['--latency-budget'],
                field_type=ConfigPrimitive(int),
                description=
                "Shorthand flag for specifying a maximum latency in ms."))

        self._add_config(
            ConfigField(
                'min_throughput',
                flags=['--min-throughput'],
                field_type=ConfigPrimitive(int),
                description="Shorthand flag for specifying a minimum throughput."
            ))

    def set_config_values(self, args: argparse.Namespace) -> None:
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
        if args.mode == 'online' and 'latency_budget' not in args:
            self._fields['objectives'].set_default_value(
                DEFAULT_ONLINE_OBJECTIVES)

        super().set_config_values(args)

        # Add plot configs and after config parse. Users should not be
        # able to edit these plots.
        self._add_plot_configs()
        if args.mode == 'online':
            self._fields['plots'].set_value(DEFAULT_ONLINE_PLOTS)
        elif args.mode == 'offline':
            self._fields['plots'].set_value(DEFAULT_OFFLINE_PLOTS)

    def _add_plot_configs(self):
        """
        Add plots to the config
        """
        plots_scheme = ConfigObject(schema={
            '*':
                ConfigObject(
                    schema={
                        'title': ConfigPrimitive(type_=str),
                        'x_axis': ConfigPrimitive(type_=str),
                        'y_axis': ConfigPrimitive(type_=str),
                        'monotonic': ConfigPrimitive(type_=bool)
                    })
        },
                                    output_mapper=ConfigPlot.from_object)
        self._add_config(
            ConfigField(
                'plots',
                field_type=ConfigUnion([
                    plots_scheme,
                    ConfigListGeneric(type_=plots_scheme,
                                      output_mapper=ConfigPlot.from_list)
                ]),
                description=
                'Model analyzer uses the information in this section to construct plots of the results.'
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

        if self.triton_launch_mode != 'docker':
            if self.triton_docker_args:
                logger.warning(
                    "Triton launch mode is not set to docker. Model Analyzer cannot set "
                    "triton_docker_args.")
        # If run config search is disabled and no concurrency or request rate is provided,
        # set the default value.
        if self.run_config_search_disable:
            if len(self.concurrency) == 0 and len(self.request_rate) == 0:
                self.concurrency = [1]

        # Change default RCS mode to quick for multi-model concurrent profiling
        if self.run_config_profile_models_concurrently_enable:
            self.run_config_search_mode = 'quick'

        if not self.export_path:
            logger.warning(
                f"--export-path not specified. Using {self._fields['export_path'].default_value()}"
            )
        elif os.path.exists(
                self.export_path) and not os.path.isdir(self.export_path):
            raise TritonModelAnalyzerException(
                f"Export path {self.export_path} is not a directory.")
        elif not os.path.exists(self.export_path):
            os.makedirs(self.export_path)

        if self.num_top_model_configs > 0 and not self.constraints:
            raise TritonModelAnalyzerException(
                "If setting num_top_model_configs > 0, comparison across models is requested. "
                "This requires that global constraints be specified in the config to be used as default."
            )

    def _autofill_values(self):
        """
        Fill in the implied or default
        config values.
        """
        cpu_only = False
        if len(self.gpus) == 0 or not cuda.is_available():
            cpu_only = True

        # Set global constraints if latency budget is specified
        if self.latency_budget:
            if self.constraints:
                constraints = self.constraints
                constraints['perf_latency_p99'] = {'max': self.latency_budget}
                if 'perf_latency' in constraints:
                    # In case a tighter perf_latency is provided
                    constraints['perf_latency'] = constraints[
                        'perf_latency_p99']
                self._fields['constraints'].set_value(constraints)
            else:
                self._fields['constraints'].set_value(
                    {'perf_latency_p99': {
                        'max': self.latency_budget
                    }})

        # Set global constraints if minimum throughput is specified
        if self.min_throughput:
            if self.constraints:
                constraints = self.constraints
                constraints['perf_throughput'] = {'min': self.min_throughput}
                self._fields['constraints'].set_value(constraints)
            else:
                self._fields['constraints'].set_value(
                    {'perf_throughput': {
                        'min': self.min_throughput
                    }})

        # Switch default output fields if request rate is being used
        # and the user didn't specify a custom output field
        if self._using_request_rate():
            if not self._fields['inference_output_fields'].is_set_by_user():
                self.inference_output_fields = DEFAULT_REQUEST_RATE_INFERENCE_OUTPUT_FIELDS

            if not self._fields['gpu_output_fields'].is_set_by_user():
                self.gpu_output_fields = DEFAULT_REQUEST_RATE_GPU_OUTPUT_FIELDS

        new_profile_models = {}
        for i, model in enumerate(self.profile_models):
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
                new_model['constraints'] = model.constraints().to_dict()

            # Weighting
            if not model.weighting():
                if 'weighting' in self._fields and self.weighting:
                    raise TritonModelAnalyzerException(
                        'Weighting can not be specified as a global parameter. Please make this a model parameter.'
                    )
                else:
                    new_model['weighting'] = DEFAULT_MODEL_WEIGHTING
            else:
                new_model['weighting'] = model.weighting()

            # Shorthands
            if self.latency_budget:
                if 'constraints' in new_model:
                    new_model['constraints']['perf_latency_p99'] = {
                        'max': self.latency_budget
                    }
                    if 'perf_latency' in new_model['constraints']:
                        # In case a tighter perf_latency is provided
                        new_model['constraints']['perf_latency'] = new_model[
                            'constraints']['perf_latency_p99']
                else:
                    new_model['constraints'] = {
                        'perf_latency_p99': {
                            'max': self.latency_budget
                        }
                    }

            if self.min_throughput:
                if 'constraints' in new_model:
                    new_model['constraints']['perf_throughput'] = {
                        'min': self.min_throughput
                    }
                else:
                    new_model['constraints'] = {
                        'perf_throughput': {
                            'min': self.min_throughput
                        }
                    }

            # Run parameters
            if not model.parameters():
                new_model['parameters'] = {
                    'batch_sizes': self.batch_sizes,
                    'concurrency': self.concurrency,
                    'request_rate': self.request_rate
                }
            else:
                new_model['parameters'] = {}
                if 'batch_sizes' in model.parameters():
                    new_model['parameters'].update(
                        {'batch_sizes': model.parameters()['batch_sizes']})
                else:
                    new_model['parameters'].update(
                        {'batch_sizes': self.batch_sizes})

                if 'concurrency' in model.parameters():
                    new_model['parameters'].update(
                        {'concurrency': model.parameters()['concurrency']})
                else:
                    new_model['parameters'].update(
                        {'concurrency': self.concurrency})

                if 'request_rate' in model.parameters():
                    new_model['parameters'].update(
                        {'request_rate': model.parameters()['request_rate']})
                else:
                    new_model['parameters'].update(
                        {'request_rate': self.request_rate})

            if new_model['parameters']['request_rate'] and new_model[
                    'parameters']['concurrency']:
                raise TritonModelAnalyzerException(
                    "Cannot specify both concurrency and request rate as model parameters."
                )

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

            # triton docker args
            if not model.triton_docker_args():
                new_model['triton_docker_args'] = self.triton_docker_args
            else:
                new_model['triton_docker_args'] = model.triton_docker_args()

            # Transfer model config parameters directly
            if model.model_config_parameters():
                new_model[
                    'model_config_parameters'] = model.model_config_parameters(
                    )

            new_profile_models[model.model_name()] = new_model
        self._fields['profile_models'].set_value(new_profile_models)

    def _using_request_rate(self) -> bool:
        if self.request_rate or self.request_rate_search_enable:
            return True
        elif self._fields['run_config_search_max_request_rate'].is_set_by_user() or \
             self._fields['run_config_search_min_request_rate'].is_set_by_user():
            return True
        else:
            return self._are_models_using_request_rate()

    def _are_models_using_request_rate(self) -> bool:
        model_using_request_rate = False
        model_using_concurrency = False
        for i, model in enumerate(self.profile_models):
            if model.parameters() and 'request_rate' in model.parameters():
                model_using_request_rate = True
            else:
                model_using_concurrency = True

        if model_using_request_rate and model_using_concurrency:
            raise TritonModelAnalyzerException("Parameters in all profiled models must use request-rate-range. "\
                "Model Analyzer does not support mixing concurrency-range and request-rate-range.")
        else:
            return model_using_request_rate

    def is_request_rate_specified(self, model_parameters: dict) -> bool:
        """
        Returns true if either the model or the config specified request rate
        """
        return 'request_rate' in model_parameters and model_parameters['request_rate'] or \
            self.request_rate_search_enable or \
            self.get_config()['run_config_search_min_request_rate'].is_set_by_user() or \
            self.get_config()['run_config_search_max_request_rate'].is_set_by_user()