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

from itertools import product
from typing import Dict, List


class GeneratorUtils:
    ''' Class for utility functions for Generators  '''

    @staticmethod
    def generate_combinations(value: object) -> List:
        """
        Generates all the alternative fields for
        a given value.

        Parameters
        ----------
        value : object
            The value to be used for sweeping.

        Returns
        -------
        list
            A list of all the alternatives for the parameters.
        """

        if type(value) is dict:
            sweeped_dict = {}
            for key, sweep_choices in value.items():
                sweep_parameter_list = []

                # This is the list of sweep parameters. When parsing a
                # config every sweepable parameter will be converted
                # to a list of values to make the parameter sweeping easier in
                # here.
                for sweep_choice in sweep_choices:
                    sweep_parameter_list += GeneratorUtils.generate_combinations(
                        sweep_choice)

                sweeped_dict[key] = sweep_parameter_list

            # Generate parameter combinations for this field.
            return GeneratorUtils.generate_parameter_combinations(sweeped_dict)

        # When this line of code is executed the value for this field is
        # a list. This list does NOT represent possible sweep values.
        # Because of this we need to ensure that in every sweep configuration,
        # one item from every list item exists.
        elif type(value) is list:

            # This list contains a set of lists. The return value from this
            # branch of the code is a list of lists where in each inner list
            # there is one item from every list item.
            sweep_parameter_list = []
            for item in value:
                sweep_parameter_list_item = GeneratorUtils.generate_combinations(
                    item)
                sweep_parameter_list.append(sweep_parameter_list_item)

            # Cartesian product of all the elements in the sweep_parameter_list
            return [list(x) for x in list(product(*sweep_parameter_list))]

        # In the default case return a list of the value. This function should
        # always return a list.
        return [value]

    @staticmethod
    def generate_parameter_combinations(params: Dict) -> List[Dict]:
        """
        Generate a list of all possible subdictionaries
        from given dictionary. The subdictionaries will
        have all the same keys, but only one value from
        each key.

        Parameters
        ----------
        params : dict
            keys are strings and the values must be lists
        """

        param_combinations = list(product(*tuple(params.values())))
        return [dict(zip(params.keys(), vals)) for vals in param_combinations]

    @staticmethod
    def generate_doubled_list(min_value: int, max_value: int) -> List[int]:
        """
        Generates a list of values from min_value -> max_value doubling 
        min_value for each entry

        Parameters
        ----------
        min_value: int
            The minimum value for the generated list
        max_value : int
            The value that the generated list will not exceed
        """

        list = []
        val = 1 if min_value == 0 else min_value
        while val <= max_value:
            list.append(val)
            val *= 2
        return list
