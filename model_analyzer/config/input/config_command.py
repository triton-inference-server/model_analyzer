# Copyright (c) 2021-22, NVIDIA CORPORATION. All rights reserved.
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

from typing import Dict, List, Optional, Any
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
import yaml
from argparse import Namespace
from .yaml_config_validator import YamlConfigValidator

from copy import deepcopy


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
            config = yaml.safe_load(config_file)
            return config

    def set_config_values(self, args: Namespace) -> None:
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

        yaml_config = self._load_yaml_config(args)
        self._check_for_illegal_config_settings(args, yaml_config)
        self._set_field_values(args, yaml_config)
        self._preprocess_and_verify_arguments()
        self._autofill_values()

    def _load_yaml_config(self, args: Namespace) -> Optional[Dict[str, List]]:
        if 'config_file' in args:
            yaml_config = self._load_config_file(args.config_file)
            YamlConfigValidator.validate(yaml_config)
        else:
            yaml_config = None

        return yaml_config

    def _check_for_illegal_config_settings(
            self, args: Namespace, yaml_config: Optional[Dict[str,
                                                              List]]) -> None:
        self._check_for_duplicate_profile_models_option(args, yaml_config)
        self._check_for_multi_model_incompatability(args, yaml_config)
        self._check_for_quick_search_incompatability(args, yaml_config)

    def _set_field_values(self, args: Namespace,
                          yaml_config: Optional[Dict[str, List]]) -> None:
        for key, value in self._fields.items():
            self._fields[key].set_name(key)
            config_value = self._get_config_value(key, args, yaml_config)

            if config_value:
                self._fields[key].set_value(config_value)
            elif value.default_value() is not None:
                self._fields[key].set_value(value.default_value())
            elif value.required():
                flags = ', '.join(value.flags())
                raise TritonModelAnalyzerException(
                    f'Config for {value.name()} is not specified. You need to specify it using the YAML config file or using the {flags} flags in CLI.'
                )

    def _get_config_value(
            self, key: str, args: Namespace,
            yaml_config: Optional[Dict[str, List]]) -> Optional[Any]:
        if key in args:
            return getattr(args, key)
        elif yaml_config is not None and key in yaml_config:
            return yaml_config[key]
        else:
            return None

    def _check_for_duplicate_profile_models_option(
            self, args: Namespace, yaml_config: Optional[Dict[str,
                                                              List]]) -> None:
        key_in_args = 'profile_models' in args
        key_in_yaml = yaml_config is not None and 'profile_models' in yaml_config

        if key_in_args and key_in_yaml:
            raise TritonModelAnalyzerException(
                f'\n The profile model option is specified on both '
                'the CLI (--profile-models) and in the YAML config file.'
                '\n Please remove the option from one of the locations and try again'
            )

    def _check_for_multi_model_incompatability(
            self, args: Namespace, yaml_config: Optional[Dict[str,
                                                              List]]) -> None:
        if not self._get_config_value(
                'run_config_profile_models_concurrently_enable', args,
                yaml_config):
            return

        self._check_multi_model_search_mode_incompatability(args, yaml_config)

    def _check_multi_model_search_mode_incompatability(
            self, args: Namespace, yaml_config: Optional[Dict[str,
                                                              List]]) -> None:
        if self._get_config_value('run_config_search_mode', args,
                                  yaml_config) != 'quick':
            raise TritonModelAnalyzerException(
                f'\nConcurrent profiling of models is only supported in quick search mode.'
                '\nPlease use quick search mode or disable concurrent model profiling.'
            )

    def _check_for_quick_search_incompatability(
            self, args: Namespace, yaml_config: Optional[Dict[str,
                                                              List]]) -> None:
        if self._get_config_value('run_config_search_mode', args,
                                  yaml_config) != 'quick':
            return

        self._check_no_search_disable(args, yaml_config)
        self._check_no_search_values(args, yaml_config)
        self._check_no_global_list_values(args, yaml_config)
        self._check_no_per_model_list_values(args, yaml_config)

    def _check_no_search_disable(
            self, args: Namespace, yaml_config: Optional[Dict[str,
                                                              List]]) -> None:
        if self._get_config_value('run_config_search_disable', args,
                                  yaml_config):
            raise TritonModelAnalyzerException(
                f'\nDisabling of run config search is not supported in quick search mode.'
                '\nPlease use brute search mode or remove --run-config-search-disable.'
            )

    def _check_no_search_values(self, args: Namespace,
                                yaml_config: Optional[Dict[str, List]]) -> None:
        max_concurrency = self._get_config_value(
            'run_config_search_max_concurrency', args, yaml_config)
        min_concurrency = self._get_config_value(
            'run_config_search_min_concurrency', args, yaml_config)
        max_instance = self._get_config_value(
            'run_config_search_max_instance_count', args, yaml_config)
        min_instance = self._get_config_value(
            'run_config_search_min_instance_count', args, yaml_config)
        max_batch_size = self._get_config_value(
            'run_config_search_max_model_batch_size', args, yaml_config)
        min_batch_size = self._get_config_value(
            'run_config_search_min_model_batch_size', args, yaml_config)

        if max_concurrency or min_concurrency:
            raise TritonModelAnalyzerException(
                f'\nProfiling of models in quick search mode is not supported with min/max concurrency search values.'
                '\nPlease use brute search mode or remove concurrency search values.'
            )
        if max_instance or min_instance:
            raise TritonModelAnalyzerException(
                f'\nProfiling of models in quick search mode is not supported with min/max instance search values.'
                '\nPlease use brute search mode or remove instance search values.'
            )
        if max_batch_size or min_batch_size:
            raise TritonModelAnalyzerException(
                f'\nProfiling of models in quick search mode is not supported with min/max batch size search values.'
                '\nPlease use brute search mode or remove batch size search values.'
            )

    def _check_no_global_list_values(
            self, args: Namespace, yaml_config: Optional[Dict[str,
                                                              List]]) -> None:
        concurrency = self._get_config_value('concurrency', args, yaml_config)
        batch_sizes = self._get_config_value('batch_sizes', args, yaml_config)

        if concurrency or batch_sizes:
            raise TritonModelAnalyzerException(
                f'\nProfiling of models in quick search mode is not supported with lists of concurrencies or batch sizes.'
                '\nPlease use brute search mode or remove concurrency/batch sizes list.'
            )

    def _check_no_per_model_list_values(
            self, args: Namespace, yaml_config: Optional[Dict[str,
                                                              List]]) -> None:
        profile_models = self._get_config_value('profile_models', args,
                                                yaml_config)

        if not profile_models or type(profile_models) is str or type(
                profile_models) is list:
            return

        for model in profile_models.values():
            if not 'parameters' in model:
                continue

            if 'concurrency' in model['parameters'] or 'batch size' in model[
                    'parameters']:
                raise TritonModelAnalyzerException(
                    f'\nProfiling of models in quick search mode is not supported with lists of concurrencies or batch sizes.'
                    '\nPlease use brute search mode or remove concurrency/batch sizes list.'
                )

        for model in profile_models.values():
            if not 'model_config_parameters' in model:
                continue

            if 'max_batch_size' in model['model_config_parameters']:
                raise TritonModelAnalyzerException(
                    f'\nProfiling of models in quick search mode is not supported with lists max batch sizes.'
                    '\nPlease use brute search mode or remov max batch size list.'
                )

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

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __getattr__(self, name):
        return self._fields[name].value()

    def __setattr__(self, name, value):
        if name == '_fields':
            self.__dict__[name] = value
        else:
            self._fields[name].set_value(value)
