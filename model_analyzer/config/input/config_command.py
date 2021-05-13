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
import yaml


class ConfigCommand:
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
        """
        Makes calls to _add_config,
        must be overloaded by subclass
        """

        raise NotImplementedError

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
            self._fields[key].set_name(key)
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
        self._preprocess_and_verify_arguments()
        self._autofill_values()

    def _preprocess_and_verify_arguments(self):
        """
        Enforces some rules on the config.

        Raises
        ------
        TritonModelAnalyzerException
            If there is a problem with arguments or config.
        """

        pass

    def _autofill_values(self):
        """
        Enforces some rules on the config.

        Raises
        ------
        TritonModelAnalyzerException
            If there is a problem with arguments or config.
        """

        pass

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
