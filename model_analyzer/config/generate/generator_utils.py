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
from math import log2


class GeneratorUtils:
    ''' Class for utility functions for Generators  '''

    @staticmethod
    def generate_combinations(value):
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

    def generate_parameter_combinations(params):
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

    def generate_log2_list(value):
        """
        Generates a list of all 2^n numbers, where 2^n does not exceed value 
        
        Parameters
        ----------
        value : int
            The value that the generated list will not exceed
        """

        log_value = int(log2(value))
        return [2**c for c in range(0, log_value + 1)]
