# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

from .common import test_result_collector as trc
from .common.test_utils import convert_to_bytes, ROOT_DIR
from .mocks.mock_config import MockConfig

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_analyze \
    import ConfigCommandAnalyze

from model_analyzer.plots.plot_manager import PlotManager
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager

from filecmp import cmp
from shutil import rmtree
from unittest.mock import MagicMock, patch


class TestPlotManager(trc.TestResultCollector):

    def setUp(self):
        self._create_single_model_result_manager()

    def tearDown(self):
        patch.stopall()

    def test_single_model_summary_plots_against_golden(self):
        """
        Match the summary plots against the golden versions in
        tests/common/single-model-ckpt
        """
        plot_manager = PlotManager(
            config=self._single_model_config,
            result_manager=self._single_model_result_manager)

        plot_manager.create_summary_plots()
        plot_manager.export_summary_plots()

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/single-model-ckpt/plots/simple/add_sub/gpu_mem_v_latency.png",
                f"{ROOT_DIR}/single-model-ckpt/golden_gpu_mem_v_latency.png"))

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/single-model-ckpt/plots/simple/add_sub/throughput_v_latency.png",
                f"{ROOT_DIR}/single-model-ckpt/golden_throughput_v_latency.png")
        )

        rmtree(f"{ROOT_DIR}/single-model-ckpt/plots/")

    def _evaluate_config(self, args, yaml_content):
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = ConfigCommandAnalyze()
        cli = CLI()
        cli.add_subcommand(
            cmd='analyze',
            help=
            'Collect and sort profiling results and generate data and summaries.',
            config=config)
        cli.parse()
        mock_config.stop()
        return config

    def _create_single_model_result_manager(self):
        args = [
            'model-analyzer', 'analyze', '-f', 'config.yml',
            '--checkpoint-directory', f'{ROOT_DIR}/single-model-ckpt/',
            '--export-path', f'{ROOT_DIR}/single-model-ckpt/'
        ]
        yaml_content = convert_to_bytes("""
            analysis_models: add_sub
        """)
        config = self._evaluate_config(args, yaml_content)
        state_manager = AnalyzerStateManager(config=config, server=None)
        state_manager.load_checkpoint(checkpoint_required=True)

        self._single_model_result_manager = ResultManager(
            config=config, state_manager=state_manager)

        self._single_model_config = config

        self._single_model_result_manager.create_tables()
        self._single_model_result_manager.compile_and_sort_results()


if __name__ == "__main__":
    unittest.main()
