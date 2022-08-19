# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tests.common.test_utils import convert_to_bytes
from tests.mocks.mock_config import MockConfig
from tests.mocks.mock_model_config import MockModelConfig
from model_analyzer.cli.cli import CLI
from config_command_experiment import ConfigCommandExperiment


class ExperimentConfigCommandCreator:
    """
    Static class to fake out the CLI/YAML process and create a ConfigCommandProfile object
    """

    @staticmethod
    def make_config(data_path, model_name, other_args):
        mock_model_config = MockModelConfig("")
        mock_model_config.start()

        checkpoint_dir = f"{data_path}/{model_name}"

        #yapf: disable
        args = [
            'model-analyzer', 'profile',
            '--profile-models', model_name,
            '--model-repository', data_path,
            '--checkpoint-directory', checkpoint_dir,
            '-f', 'path-to-config-file'
        ]
        args += other_args

        yaml_content = convert_to_bytes("")

        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = ConfigCommandExperiment()
        cli = CLI()
        cli.add_subcommand(
            cmd='profile',
            help='Run model inference profiling based on specified CLI or '
            'config options.',
            config=config)
        cli.parse()
        mock_config.stop()

        mock_model_config.stop()
        return config
