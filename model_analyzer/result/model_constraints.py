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


class ModelConstraints:
    """
    A class representing the Constraints used for a single model.
    """

    def __init__(self, constraints):
        """
        Parameters
        ----------
        constraints: dict
            keys are strings and 
            values are dict of single str: int pair
        """

        self._constraints = {}
        if constraints:
            self._constraints = constraints

    def to_dict(self):
        """
        Returns constraints dictionary

        Returns
        ----------
        constraints: dict
        """
        return self._constraints

    def has_metric(self, name):
        """
        To check if given metric tag is present in constraints

        Returns
        ----------
        bool:
            True if metric is present in constraints else False
        """
        if name and name in self._constraints:
            return True
        else:
            return False

    def __getitem__(self, name):
        """
        To subscript constraints using metric name
        ex: model_constraints['perf_latency_p99']

        Parameters
        ----------
        name: str
            metric name
        """
        if name in self._constraints:
            return self._constraints[name]
        else:
            msg = f"'{name}' key not found in constraints"
            raise KeyError(msg)

    def __bool__(self):
        """
        To check if constraints are empty
        """
        if self._constraints:
            return True
        else:
            return False

    def __eq__(self, other):
        """
        To compare two ModelConstraints objects

        Parameters
        ----------
        other: ModelConstraints object
        """
        if self._constraints == other._constraints:
            return True
        else:
            return False

    def items(self):
        return self._constraints.items()

    def __repr__(self):
        return str(self._constraints)

    def __iter__(self):
        return iter(self._constraints)
