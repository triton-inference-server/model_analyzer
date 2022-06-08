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

from model_analyzer.result.constraint_manager import ConstraintManager

from .mocks.mock_config import MockConfig
from model_analyzer.config.input.config_command_analyze import ConfigCommandAnalyze
from model_analyzer.cli.cli import CLI

from .common.test_utils import convert_to_bytes

import unittest
from unittest.mock import MagicMock, patch
from .common import test_result_collector as trc


class TestConstraintManager(trc.TestResultCollector):

    def setUp(self):
        NotImplemented

    def tearDown(self):
        patch.stopall()

    def test_single_model_no_constraints(self):
        """
        Test that constraints are empty
        """
        config = self._create_single_model_no_constraints()
        constraints = ConstraintManager.get_constraints_for_all_models(config)

        self.assertEqual(constraints['model_A'], None)
        self.assertEqual(constraints['default'], {})

    def test_single_model_with_constraints(self):
        """
        Test that model specific constraints are set
        """
        config = self._create_single_model_with_constraints()
        constraints = ConstraintManager.get_constraints_for_all_models(config)

        self.assertEqual(constraints['model_A'],
                         {'perf_latency_p99': {
                             'max': 100
                         }})
        self.assertEqual(constraints['default'], {})

    def test_single_model_with_global_constraints(self):
        """
        Test that global constraints are attributed to a model
        """
        config = self._create_single_model_global_constraints()
        constraints = ConstraintManager.get_constraints_for_all_models(config)

        self.assertEqual(constraints['model_A'],
                         {'perf_latency_p99': {
                             'max': 100
                         }})
        self.assertEqual(constraints['default'],
                         {'perf_latency_p99': {
                             'max': 100
                         }})

    def test_single_model_with_both_constraints(self):
        """
        Test that model specific constraints override global
        """
        config = self._create_single_model_both_constraints()
        constraints = ConstraintManager.get_constraints_for_all_models(config)

        self.assertEqual(constraints['model_A'],
                         {'perf_latency_p99': {
                             'max': 50
                         }})
        self.assertEqual(constraints['default'],
                         {'perf_latency_p99': {
                             'max': 100
                         }})

    def test_multi_model_with_both_constraints(self):
        """
        Test multi-model with both styles of constraints
        """
        config = self._create_multi_model_both_constraints()
        constraints = ConstraintManager.get_constraints_for_all_models(config)

        self.assertEqual(constraints['model_A'],
                         {'perf_latency_p99': {
                             'max': 50
                         }})
        self.assertEqual(constraints['model_B'],
                         {'perf_throughput': {
                             'min': 100
                         }})
        self.assertEqual(constraints['model_C'], {
            'gpu_used_memory': {
                'max': 50
            },
            'perf_throughput': {
                'min': 50
            }
        })
        self.assertEqual(constraints['model_D'], {
            'perf_latency_p99': {
                'max': 100
            },
            'gpu_used_memory': {
                'max': 100
            }
        })

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

    def _create_single_model_no_constraints(self):
        args = self._create_args()
        yaml_content = convert_to_bytes("""
            analysis_models: 
              model_A
        """)
        config = self._evaluate_config(args, yaml_content)

        return config

    def _create_single_model_with_constraints(self):
        args = self._create_args()
        yaml_content = convert_to_bytes("""
            analysis_models: 
              model_A:
                constraints:
                  perf_latency_p99:
                    max: 100
        """)
        config = self._evaluate_config(args, yaml_content)

        return config

    def _create_single_model_global_constraints(self):
        args = self._create_args()
        yaml_content = convert_to_bytes("""
            analysis_models: 
              model_A
                
            constraints:
                perf_latency_p99:
                  max: 100
        """)
        config = self._evaluate_config(args, yaml_content)

        return config

    def _create_single_model_both_constraints(self):
        args = self._create_args()
        yaml_content = convert_to_bytes("""
            analysis_models: 
              model_A:
                constraints:
                  perf_latency_p99:
                    max: 50
                
            constraints:
                perf_latency_p99:
                  max: 100
        """)
        config = self._evaluate_config(args, yaml_content)

        return config

    def _create_multi_model_both_constraints(self):
        args = self._create_args()
        yaml_content = convert_to_bytes("""
            analysis_models:
              model_A:
                constraints:
                  perf_latency_p99:
                    max: 50
              model_B:
                constraints:
                  perf_throughput:
                    min: 100
              model_C:
                constraints:
                  gpu_used_memory:
                    max: 50
                  perf_throughput:
                    min: 50
              model_D:
                objectives:
                  perf_throughput  
                    
            constraints:
                perf_latency_p99:
                  max: 100
                gpu_used_memory:
                  max: 100
        """)
        config = self._evaluate_config(args, yaml_content)

        return config

    def _create_args(self):
        return [
            'model-analyzer',
            'analyze',
            '-f',
            'config.yml',
        ]


if __name__ == '__main__':
    unittest.main()