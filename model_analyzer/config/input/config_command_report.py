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

from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from .config_union import ConfigUnion
from .config_object import ConfigObject
from .config_enum import ConfigEnum
from .config_list_generic import ConfigListGeneric
from .config_list_string import ConfigListString
from .config_defaults import \
    DEFAULT_CHECKPOINT_DIRECTORY, DEFAULT_EXPORT_PATH, \
    DEFAULT_OFFLINE_REPORT_PLOTS, DEFAULT_ONLINE_REPORT_PLOTS, DEFAULT_REPORT_FORMAT
from .config_field import ConfigField
from .config_primitive import ConfigPrimitive
from .config_command import ConfigCommand

from .objects.config_plot import ConfigPlot
from .objects.config_model_report_spec import ConfigModelReportSpec

import logging
import os


class ConfigCommandReport(ConfigCommand):
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
                flags=['-f', '--config-file'],
                field_type=ConfigPrimitive(str),
                description="Path to Config File for subcommand 'report'."))
        self._add_config(
            ConfigField(
                'checkpoint_directory',
                flags=['--checkpoint-directory', '-s'],
                default_value=DEFAULT_CHECKPOINT_DIRECTORY,
                field_type=ConfigPrimitive(str),
                description=
                "Full path to directory to which to read and write checkpoints and profile data."
            ))
        self._add_config(
            ConfigField(
                'export_path',
                flags=['--export-path', '-e'],
                default_value=DEFAULT_EXPORT_PATH,
                field_type=ConfigPrimitive(str),
                description=
                "Full path to directory in which to store the results"))
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
                default_value=DEFAULT_ONLINE_REPORT_PLOTS,
                description=
                'Model analyzer uses the information in this section to construct plots of the results.'
            ))

        report_model_scheme = ConfigObject(
            required=True,
            schema={
                # Any key is allowed, but the keys must follow the pattern
                # below
                '*': ConfigObject(schema={'plots': plots_scheme})
            },
            output_mapper=ConfigModelReportSpec.
            model_object_to_config_model_report_spec)
        self._add_config(
            ConfigField(
                'report_model_configs',
                flags=['--report-model-configs', '-n'],
                field_type=ConfigUnion([
                    report_model_scheme,
                    ConfigListGeneric(ConfigUnion([
                        report_model_scheme,
                        ConfigPrimitive(str,
                                        output_mapper=ConfigModelReportSpec.
                                        model_str_to_config_model_report_spec)
                    ]),
                                      required=True,
                                      output_mapper=ConfigModelReportSpec.
                                      model_mixed_to_config_model_report_spec),
                    ConfigListString(output_mapper=ConfigModelReportSpec.
                                     model_list_to_config_model_report_spec),
                ],
                                       required=True),
                description=(
                    'Comma delimited list of the names of model configs'
                    ' for which to generate detailed reports.')))
        self._add_config(
            ConfigField('output_formats',
                        flags=['--output-formats', '-o'],
                        default_value=DEFAULT_REPORT_FORMAT,
                        field_type=ConfigUnion([
                            ConfigListGeneric(type_=ConfigEnum(
                                choices=['pdf', 'csv', 'png'])),
                            ConfigListString()
                        ]),
                        description='Output file format for detailed report.'))

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

        if args.mode == 'online':
            self._fields['plots'].set_default_value(
                DEFAULT_ONLINE_REPORT_PLOTS)
        elif args.mode == 'offline':
            self._fields['plots'].set_default_value(
                DEFAULT_OFFLINE_REPORT_PLOTS)

        super().set_config_values(args)

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

    def _autofill_values(self):
        """
        Fill in the implied or default
        config values.
        """

        new_report_model_configs = {}
        for model in self.report_model_configs:
            new_report_model_config = {}

            # Plots
            if not model.plots():
                new_report_model_config['plots'] = {
                    plot.name(): {
                        'title': plot.title(),
                        'x_axis': plot.x_axis(),
                        'y_axis': plot.y_axis(),
                        'monotonic': plot.monotonic()
                    }
                    for plot in self.plots
                }
            else:
                new_report_model_config['plots'] = {
                    plot.name(): {
                        'title': plot.title(),
                        'x_axis': plot.x_axis(),
                        'y_axis': plot.y_axis(),
                        'monotonic': plot.monotonic()
                    }
                    for plot in model.plots()
                }

            new_report_model_configs[
                model.model_config_name()] = new_report_model_config

        self._fields['report_model_configs'].set_value(
            new_report_model_configs)
