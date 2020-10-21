# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from itertools import product
from subprocess import check_output, CalledProcessError, STDOUT

from model_analyzer.analyzer.perf_analyzer.perf_record import PerfRecord
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class PerfAnalyzer:
    """
    This class provides an interface for running workloads
    with perf_analyzer.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : PerfAnalyzerConfig
            keys are names of arguments to perf_analyzer,
            values are their values.
        """
        self._config = config
        self._outputs = []

    def run_job(self, sweep_params):
        """
        Parses the parameters to sweep over and runs
        the perf_analyzer once per configuration.

        Parameters
        ----------
        sweep_params : dictionary of lists
            keys are arguments to perf_analyzer, values are
            list of argument values to run perf_analyzer with.

        Returns
        -------
        List of tuples
            (params, output) where params are the set of
            parameters provided for that run and output is
            the PerfRecord for the perf_analyzer run

        Raises
        ------
        TritonModelAnalyzerException
            If _run_perf_analyzer throws this exception
        """
        # Create a config for each combination of parameters
        combos = list(product(*tuple(sweep_params.values())))
        run_configs = [dict(zip(sweep_params.keys(), vals)) for vals in combos]

        # Collect a tuple (run_config, output)
        run_outputs = []

        # Run parameters are some subset of perf_analyzer config
        for run_config in run_configs:

            # update perf_analyzer config
            for key, val in run_config.items():
                self._config[key] = val

            # Run perf_analyzer
            run_record = self._run_perf_analyzer()
            run_outputs.append((run_config, run_record))

        return run_outputs

    def _run_perf_analyzer(self):
        """
        Runs the perf_analyzer tool locally
        and synchronously

        Returns
        -------
        PerfRecord
            output from the subprocess call to
            the perf_analyzer

        Raises
        ------
        TritonModelAnalyzerException
            If subprocess throws CalledProcessError
        """
        cmd = ['perf_analyzer']
        cmd += self._config.to_cli_string().replace('=', ' ').split()

        # Synchronously start and finish run

        try:
            out = check_output(cmd, stderr=STDOUT, encoding='utf-8')
        except CalledProcessError as e:
            raise TritonModelAnalyzerException(
                f"perf analyzer returned with exit"
                " status {e.returncode} : {e.output}")

        return PerfRecord(out)
