# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

from model_analyzer.config.run.model_run_config import ModelRunConfig


class RunConfig:
    """
    Encapsulates all the information needed to run one or more models 
    at the same time in Perf Analyzer
    """

    def __init__(self, triton_env):
        """
        Parameters
        ----------
        triton_env : dict
            A dictionary of environment variables to set
            when launching tritonserver
        """

        self._triton_env = triton_env
        self._model_run_configs = []

    def add_model_run_config(self, model_run_config):
        """
        Add a ModelRunConfig to this RunConfig
        """
        self._model_run_configs.append(model_run_config)

    def model_run_configs(self):
        """
        Returns the list of ModelRunConfigs to run concurrently
        """
        return self._model_run_configs

    def representation(self):
        """
        Returns a representation string for the RunConfig that can be used
        as a key to uniquely identify it
        """
        return ''.join(
            [mrc.representation() for mrc in self.model_run_configs()])

    def is_legal_combination(self):
        """
        Returns true if all model_run_configs are valid
        """
        return all([
            model_run_config.is_legal_combination()
            for model_run_config in self._model_run_configs
        ])

    def is_ensemble_model(self) -> bool:
        """
        Returns true if the first model config is an ensemble
        (an ensemble cannot be part of a multi-model)
        """
        return self._model_run_configs[0].is_ensemble_model()

    def is_bls_model(self) -> bool:
        """
        Returns true if the first model config is a BLS model
        (a BLS cannot be part of a multi-model)
        """
        return self._model_run_configs[0].is_bls_model()

    def cpu_only(self):
        """
        Returns true if all model_run_configs only operate on the CPU
        """
        return all([
            model_run_config.model_config().cpu_only()
            for model_run_config in self._model_run_configs
        ])

    def triton_environment(self):
        """
        Returns
        -------
        dict
            The environment that tritonserver
            was run with for this RunConfig
        """

        return self._triton_env

    def models_name(self):
        """ Returns a single comma-joined name of the original model names """
        return ','.join([mrc.model_name() for mrc in self.model_run_configs()])

    def model_variants_name(self):
        """ Returns a single comma-joined name of the model variant names """
        return ','.join(
            [mrc.model_variant_name() for mrc in self.model_run_configs()])

    def composing_configs(self):
        """
        Returns a list of composing configs from the first model config
        (an ensemble/BLS cannot be part of a multi-model profile)
        """
        return self._model_run_configs[0].composing_configs()

    @classmethod
    def from_dict(cls, run_config_dict):
        run_config = RunConfig({})

        run_config._triton_env = run_config_dict['_triton_env']
        for mrc_dict in run_config_dict['_model_run_configs']:
            run_config._model_run_configs.append(
                ModelRunConfig.from_dict(mrc_dict))

        return run_config
