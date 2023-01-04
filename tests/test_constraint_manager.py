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
from model_analyzer.result.model_constraints import ModelConstraints

from .common.test_utils import construct_run_config_measurement, evaluate_mock_config

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

        self.assertEqual(constraints['model_A'], ModelConstraints({}))
        self.assertEqual(constraints['default'], ModelConstraints({}))

    def test_single_model_with_constraints(self):
        """
        Test that model specific constraints are set
        """
        config = self._create_single_model_with_constraints()
        constraints = ConstraintManager.get_constraints_for_all_models(config)

        self.assertEqual(constraints['model_A'],
                         ModelConstraints({'perf_latency_p99': {
                             'max': 100
                         }}))
        self.assertEqual(constraints['default'], ModelConstraints({}))

    def test_single_model_with_global_constraints(self):
        """
        Test that global constraints are attributed to a model
        """
        config = self._create_single_model_global_constraints()
        constraints = ConstraintManager.get_constraints_for_all_models(config)

        self.assertEqual(constraints['model_A'],
                         ModelConstraints({'perf_throughput': {
                             'min': 100
                         }}))
        self.assertEqual(constraints['default'],
                         ModelConstraints({'perf_throughput': {
                             'min': 100
                         }}))

    def test_single_model_with_both_constraints(self):
        """
        Test that model specific constraints override global
        """
        config = self._create_single_model_both_constraints()
        constraints = ConstraintManager.get_constraints_for_all_models(config)

        self.assertEqual(constraints['model_A'],
                         ModelConstraints({'perf_latency_p99': {
                             'max': 50
                         }}))
        self.assertEqual(constraints['default'],
                         ModelConstraints({'perf_latency_p99': {
                             'max': 100
                         }}))

    def test_multi_model_with_both_constraints(self):
        """
        Test multi-model with both styles of constraints
        """
        config = self._create_multi_model_both_constraints()
        constraints = ConstraintManager.get_constraints_for_all_models(config)

        self.assertEqual(constraints['model_A'],
                         ModelConstraints({'perf_latency_p99': {
                             'max': 50
                         }}))
        self.assertEqual(constraints['model_B'],
                         ModelConstraints({'perf_throughput': {
                             'min': 100
                         }}))
        self.assertEqual(constraints['model_C'], ModelConstraints({
            'gpu_used_memory': {
                'max': 50
            },
            'perf_throughput': {
                'min': 50
            }
        }))
        self.assertEqual(constraints['model_D'], ModelConstraints({
            'perf_latency_p99': {
                'max': 100
            },
            'gpu_used_memory': {
                'max': 100
            }
        }))

    def test_single_model_max_constraint_checks(self):
        """
        Test that satisfies_constraints works for a single model
        with a max style constraint
        """
        config = self._create_single_model_with_constraints()
        constraints = [
            ConstraintManager.get_constraints_for_all_models(config)['model_A']
        ]

        # Constraint is P99 Latency max of 100
        rcm = self._construct_rcm({"perf_latency_p99": 101})
        self.assertFalse(
            ConstraintManager.satisfies_constraints(constraints, rcm))

        rcm = self._construct_rcm({"perf_latency_p99": 100})
        self.assertTrue(
            ConstraintManager.satisfies_constraints(constraints, rcm))

        rcm = self._construct_rcm({"perf_latency_p99": 99})
        self.assertTrue(
            ConstraintManager.satisfies_constraints(constraints, rcm))

    def test_single_model_min_constraint_checks(self):
        """
        Test that satisfies_constraints works for a single model
        with a min style constraint
        """
        config = self._create_single_model_global_constraints()
        constraints = [
            ConstraintManager.get_constraints_for_all_models(config)['model_A']
        ]

        # Constraint is throughput min of 100
        rcm = self._construct_rcm({"perf_throughput": 101})
        self.assertTrue(
            ConstraintManager.satisfies_constraints(constraints, rcm))

        rcm = self._construct_rcm({"perf_throughput": 100})
        self.assertTrue(
            ConstraintManager.satisfies_constraints(constraints, rcm))

        rcm = self._construct_rcm({"perf_throughput": 99})
        self.assertFalse(
            ConstraintManager.satisfies_constraints(constraints, rcm))

    def test_multi_model_constraint_checks(self):
        """
        Test that satisfies_constraints works for a multi model
        """
        config = self._create_multi_model_both_constraints()
        mm_constraints_dict = ConstraintManager.get_constraints_for_all_models(
            config)
        constraints = [
            mm_constraints_dict['model_A'], mm_constraints_dict['model_B']
        ]

        # Constraints are:
        #  Model A: P99 Latency max of 50
        #  Model B: Throughput min of 100

        # Model A & B are both at boundaries
        rcm = self._construct_mm_rcm([{
            "perf_latency_p99": 50,
            "perf_throughput": 0
        }, {
            "perf_latency_p99": 0,
            "perf_throughput": 100
        }])
        self.assertTrue(
            ConstraintManager.satisfies_constraints(constraints, rcm))

        # Model A exceeds latency
        rcm = self._construct_mm_rcm([{
            "perf_latency_p99": 51,
            "perf_throughput": 0
        }, {
            "perf_latency_p99": 0,
            "perf_throughput": 100
        }])
        self.assertFalse(
            ConstraintManager.satisfies_constraints(constraints, rcm))

        # Model B doesn't have enough throughput
        rcm = self._construct_mm_rcm([{
            "perf_latency_p99": 50,
            "perf_throughput": 0
        }, {
            "perf_latency_p99": 0,
            "perf_throughput": 99
        }])
        self.assertFalse(
            ConstraintManager.satisfies_constraints(constraints, rcm))

    def test_single_model_max_failure_percentage(self):
        """
        Test that constraint_failure_percentage works for a single model
        with a max style constraint
        """
        config = self._create_single_model_with_constraints()
        constraints = [
            ConstraintManager.get_constraints_for_all_models(config)['model_A']
        ]

        # Constraint is P99 Latency max of 100
        rcm = self._construct_rcm({"perf_latency_p99": 225})
        self.assertEqual(
            ConstraintManager.constraint_failure_percentage(constraints, rcm),
            125)
        rcm = self._construct_rcm({"perf_latency_p99": 150})
        self.assertEqual(
            ConstraintManager.constraint_failure_percentage(constraints, rcm),
            50)

        rcm = self._construct_rcm({"perf_latency_p99": 100})
        self.assertEqual(
            ConstraintManager.constraint_failure_percentage(constraints, rcm),
            0)

        rcm = self._construct_rcm({"perf_latency_p99": 99})
        self.assertEqual(
            ConstraintManager.constraint_failure_percentage(constraints, rcm),
            0)

    def test_single_model_min_failure_percentage(self):
        """
        Test that constraint_failure_percentage works for a single model
        with a min style constraint
        """

        config = self._create_single_model_global_constraints()
        constraints = [
            ConstraintManager.get_constraints_for_all_models(config)['model_A']
        ]

        # Constraint is throughput min of 100
        rcm = self._construct_rcm({"perf_throughput": 25})
        self.assertEqual(
            ConstraintManager.constraint_failure_percentage(constraints, rcm),
            75)
        rcm = self._construct_rcm({"perf_throughput": 50})
        self.assertEqual(
            ConstraintManager.constraint_failure_percentage(constraints, rcm),
            50)

        rcm = self._construct_rcm({"perf_throughput": 100})
        self.assertEqual(
            ConstraintManager.constraint_failure_percentage(constraints, rcm),
            0)

        rcm = self._construct_rcm({"perf_throughput": 101})
        self.assertEqual(
            ConstraintManager.constraint_failure_percentage(constraints, rcm),
            0)

    def test_multi_model_failure_percentage(self):
        """
        Test that failure percentage works for a multi model setup
        """
        config = self._create_multi_model_both_constraints()
        mm_constraints_dict = ConstraintManager.get_constraints_for_all_models(
            config)
        constraints = [
            mm_constraints_dict['model_A'], mm_constraints_dict['model_B']
        ]

        # Constraints are:
        #  Model A: P99 Latency max of 50
        #  Model B: Throughput min of 100

        # Model A & B are both at boundaries
        rcm = self._construct_mm_rcm([{
            "perf_latency_p99": 50,
            "perf_throughput": 0
        }, {
            "perf_latency_p99": 0,
            "perf_throughput": 100
        }])
        self.assertEqual(
            ConstraintManager.constraint_failure_percentage(constraints, rcm),
            0)

        # Model A exceeds latency, Model B misses on throughput - each by 20%
        rcm = self._construct_mm_rcm([{
            "perf_latency_p99": 60,
            "perf_throughput": 0
        }, {
            "perf_latency_p99": 0,
            "perf_throughput": 80
        }])
        self.assertEqual(
            ConstraintManager.constraint_failure_percentage(constraints, rcm),
            40)

        # Model A exceeds latency by 40%, Model B misses on throughput by 10%
        rcm = self._construct_mm_rcm([{
            "perf_latency_p99": 70,
            "perf_throughput": 0
        }, {
            "perf_latency_p99": 0,
            "perf_throughput": 90
        }])
        self.assertEqual(
            ConstraintManager.constraint_failure_percentage(constraints, rcm),
            50)

    def _create_single_model_no_constraints(self):
        args = self._create_args()
        yaml_str = ("""
            profile_models: 
              model_A
        """)
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def _create_single_model_with_constraints(self):
        args = self._create_args()
        yaml_str = ("""
            profile_models: 
              model_A:
                constraints:
                  perf_latency_p99:
                    max: 100
        """)
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def _create_single_model_global_constraints(self):
        args = self._create_args()
        yaml_str = ("""
            profile_models: 
              model_A
                
            constraints:
                perf_throughput:
                  min: 100
        """)
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def _create_single_model_both_constraints(self):
        args = self._create_args()
        yaml_str = ("""
            profile_models: 
              model_A:
                constraints:
                  perf_latency_p99:
                    max: 50
                
            constraints:
                perf_latency_p99:
                  max: 100
        """)
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def _create_multi_model_both_constraints(self):
        args = self._create_args()
        yaml_str = ("""
            profile_models:
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
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def _construct_rcm(self, non_gpu_metric_values):
        rcm = construct_run_config_measurement(
            model_name=MagicMock(),
            model_config_names=["test_config_name"],
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=MagicMock(),
            non_gpu_metric_values=[non_gpu_metric_values])

        return rcm

    def _construct_mm_rcm(self, non_gpu_metric_values):
        rcm = construct_run_config_measurement(
            model_name=MagicMock(),
            model_config_names=["test_config_name_A", "test_config_name_B"],
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=MagicMock(),
            non_gpu_metric_values=non_gpu_metric_values)

        return rcm

    def _create_args(self):
        return ['model-analyzer', 'profile', '-f', 'config.yml', '-m', '.']


if __name__ == '__main__':
    unittest.main()
