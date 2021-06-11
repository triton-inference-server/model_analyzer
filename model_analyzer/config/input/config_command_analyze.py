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

from .config_defaults import \
    DEFAULT_CHECKPOINT_DIRECTORY, DEFAULT_EXPORT_PATH, \
    DEFAULT_FILENAME_MODEL_GPU, DEFAULT_FILENAME_MODEL_INFERENCE, \
    DEFAULT_FILENAME_SERVER_ONLY, DEFAULT_GPU_OUTPUT_FIELDS, \
    DEFAULT_INFERENCE_OUTPUT_FIELDS, DEFAULT_NUM_CONFIGS_PER_MODEL, \
    DEFAULT_NUM_TOP_MODEL_CONFIGS, DEFAULT_OFFLINE_OBJECTIVES, DEFAULT_ONLINE_ANALYSIS_PLOTS, \
    DEFAULT_OFFLINE_ANALYSIS_PLOTS, DEFAULT_ONLINE_OBJECTIVES,DEFAULT_SERVER_OUTPUT_FIELDS, DEFAULT_SUMMARIZE_FLAG
from .config_field import ConfigField
from .config_object import ConfigObject
from .config_union import ConfigUnion
from .config_primitive import ConfigPrimitive
from .config_list_string import ConfigListString
from .config_list_generic import ConfigListGeneric
from .config_command import ConfigCommand

from .objects.config_plot import ConfigPlot
from model_analyzer.config.input.objects.config_model_analysis_spec \
    import ConfigModelAnalysisSpec
from model_analyzer.record.record import RecordType
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

import logging
import os


class ConfigCommandAnalyze(ConfigCommand):
    """
    Model Analyzer config object.
    """
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
                description="Path to config file for subcommand 'analyze'."))
        self._add_config(
            ConfigField(
                'checkpoint_directory',
                flags=['--checkpoint-directory', '-s'],
                default_value=DEFAULT_CHECKPOINT_DIRECTORY,
                field_type=ConfigPrimitive(str),
                description=
                "Full path to directory to which to read and write checkpoints and profile data."
            ))

        self._add_model_spec_configs()
        self._add_export_configs()
        self._add_report_configs()
        self._add_table_configs()
        self._add_shorthand_configs()

    def _add_model_spec_configs(self):
        """
        Adds configs related to specifying
        model objectives and constraints, 
        as well as plots
        """
        def objective_list_output_mapper(objectives):
            # Takes a list of objectives and maps them
            # into a dict
            output_dict = {}
            for objective in objectives:
                value = ConfigPrimitive(type_=int)
                value.set_value(10)
                output_dict[objective] = value
            return output_dict

        objectives_scheme = ConfigUnion([
            ConfigObject(
                schema={
                    tag: ConfigPrimitive(type_=int)
                    for tag in RecordType.get_all_record_types().keys()
                }),
            ConfigListString(output_mapper=objective_list_output_mapper)
        ])
        self._add_config(
            ConfigField(
                'objectives',
                field_type=objectives_scheme,
                default_value=DEFAULT_OFFLINE_OBJECTIVES,
                description=
                'Model Analyzer uses the objectives described here to find the best configuration for each model.'
            ))
        constraints_scheme = ConfigObject(
            schema={
                'perf_throughput':
                ConfigObject(schema={
                    'min': ConfigPrimitive(int),
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
                'constraints',
                field_type=constraints_scheme,
                description=
                'Constraints on the objectives specified in the "objectives" field of the config.'
            ))

        analysis_model_scheme = ConfigObject(
            required=True,
            schema={
                # Any key is allowed, but the keys must follow the pattern
                # below
                '*':
                ConfigObject(
                    schema={
                        'objectives': objectives_scheme,
                        'constraints': constraints_scheme,
                    })
            },
            output_mapper=ConfigModelAnalysisSpec.
            model_object_to_config_model_analysis_spec)
        self._add_config(
            ConfigField(
                'analysis_models',
                flags=['--analysis-models'],
                field_type=ConfigUnion([
                    analysis_model_scheme,
                    ConfigListGeneric(
                        ConfigUnion([
                            analysis_model_scheme,
                            ConfigPrimitive(
                                str,
                                output_mapper=ConfigModelAnalysisSpec.
                                model_str_to_config_model_analysis_spec)
                        ]),
                        required=True,
                        output_mapper=ConfigModelAnalysisSpec.
                        model_mixed_to_config_model_analysis_spec),
                    ConfigListString(output_mapper=ConfigModelAnalysisSpec.
                                     model_list_to_config_model_analysis_spec),
                ],
                                       required=True),
                description=
                'Comma-delimited list of the model names for whom to generate reports.'
            ))

    def _add_export_configs(self):
        """
        Add configs related to exporting data
        """

        self._add_config(
            ConfigField(
                'export_path',
                flags=['--export-path', '-e'],
                default_value=DEFAULT_EXPORT_PATH,
                field_type=ConfigPrimitive(str),
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
                description='Specifies filename for storing model GPU metrics')
        )
        self._add_config(
            ConfigField(
                'filename_server_only',
                flags=['--filename-server-only'],
                field_type=ConfigPrimitive(str),
                default_value=DEFAULT_FILENAME_SERVER_ONLY,
                description='Specifies filename for server-only metrics'))

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
                description='Specifies column keys for model gpu metrics table'
            ))
        self._add_config(
            ConfigField('server_output_fields',
                        flags=['--server-output-fields'],
                        field_type=ConfigListString(),
                        default_value=DEFAULT_SERVER_OUTPUT_FIELDS,
                        description=
                        'Specifies column keys for server-only metrics table'))

    def _add_report_configs(self):
        """
        Adds report related configs
        """

        self._add_config(
            ConfigField(
                'summarize',
                flags=['--summarize'],
                field_type=ConfigPrimitive(bool),
                default_value=DEFAULT_SUMMARIZE_FLAG,
                description=
                'Model Analyzer generates a brief summary of the collected data.'
            ))
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
            ConfigField('min_throughput',
                        flags=['--min-throughput'],
                        field_type=ConfigPrimitive(int),
                        description=
                        "Shorthand flag for specifying a minimum throughput."))

    def _preprocess_and_verify_arguments(self):
        """
        Enforces some rules on the config.

        Raises
        ------
        TritonModelAnalyzerException
            If there is a problem with arguments or config.
        """

        if not self.export_path:
            logging.warning(
                f"--export-path not specified. Using {self._fields['export_path'].default_value()}"
            )
        elif self.export_path and not os.path.isdir(self.export_path):
            raise TritonModelAnalyzerException(
                f"Export path {self.export_path} is not a directory.")

        if self.num_top_model_configs > 0 and not self.constraints:
            raise TritonModelAnalyzerException(
                "If setting num_top_model_configs > 0, comparison across models is requested. "
                "This requires that global constraints be specified in the config to be used as default."
            )

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

        if args.mode == 'online' and 'latency_budget' not in args:
            self._fields['objectives'].set_default_value(
                DEFAULT_ONLINE_OBJECTIVES)

        super().set_config_values(args)

        # Add plot configs and after config parse. User should not be able to edit these plots
        self._add_plot_configs()
        if args.mode == 'online':
            self._fields['plots'].set_value(DEFAULT_ONLINE_ANALYSIS_PLOTS)
        elif args.mode == 'offline':
            self._fields['plots'].set_value(DEFAULT_OFFLINE_ANALYSIS_PLOTS)

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

    def _autofill_values(self):
        """
        Fill in the implied or default
        config values.
        """

        # Set global constraints if latency budget is specified
        if self.latency_budget:
            if self.constraints:
                constraints = self.constraints
                constraints['perf_latency'] = {'max': self.latency_budget}
                self._fields['constraints'].set_value(constraints)
            else:
                self._fields['constraints'].set_value(
                    {'perf_latency': {
                        'max': self.latency_budget
                    }})

        # Set global constraints if latency budget is specified
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

        new_analysis_models = {}
        for model in self.analysis_models:
            new_model = {}

            # Objectives
            if not model.objectives():
                new_model['objectives'] = self.objectives
            else:
                new_model['objectives'] = model.objectives()
            # Constraints
            if not model.constraints():
                if 'constraints' in self._fields and self.constraints:
                    new_model['constraints'] = self.constraints
            else:
                new_model['constraints'] = model.constraints()

            # Shorthands
            if self.latency_budget:
                if 'constraints' in new_model:
                    new_model['constraints']['perf_latency'] = {
                        'max': self.latency_budget
                    }
                else:
                    new_model['constraints'] = {
                        'perf_latency': {
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
            new_analysis_models[model.model_name()] = new_model
        self._fields['analysis_models'].set_value(new_analysis_models)
