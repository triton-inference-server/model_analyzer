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

class PerfAnalyzerConfig:
    """A config class to set arguments to the perf_analyzer. 
    An argument set to None will use the perf_analyzer's default.
    """
    
    def __init__(self):
        self._args = {

            # Measurement parameters
            'async' : None,
            'sync' : None,
            'measurement-interval' : None,
            'concurrency-range' : None,
            'request-rate-range' : None,
            'request-distribution' : None,
            'request-intervals' : None,
            'binary-search' : None,
            'num-of-sequence' : None,
            'latency-threshold' : None,
            'max-threads' : None,
            'stability-percentage' : None,
            'max-trials' : None,
            'percentile' : None,
        
            'input-data' : None,
            'shared-memory' : None,
            'output-shared-memory-size' : None,
            'shape' : None,
            'sequence-length' : None,
            'string-length' : None,
            'string-data' : None,
        }

        self._options = {
            
            # options
            '-m' : None,
            '-x' : None,
            '-b' : None,
            '-u' : None,
            '-i' : None,
            '-f' : None,
            '-H' : None
        }

        self._verbose = {
            # verbose flags
            '-v' : None,
            '-v -v' : None
        }

    def to_cli_string(self):
        """Utility function to convert a config into a
        string of arguments to the perf_analyzer with CLI.
        """
        args = ['{} {}'.format(k,v) for k,v in self._options.items() if v]
        
        args += [k for k, v in self._verbose.items() if v]
        
        args += ['--{}={}'.format(k,v) for k,v in self._args.items() if v]

        return ' '.join(args)

    def __getitem__(self, key):
        return self._args[key]
    
    def __setitem__(self, key, value):
        input_to_options = {    
            'model-name' : '-m',
            'model-version' : '-x',
            'batch-size' : '-b',
            'url' : '-u',
            'protocol' : '-i',
            'latency-report-file' : '-f',
            'streaming' : '-H'
        }

        verbose = {
            'verbose' : '-v',
            'extra-verbose' : '-v -v'
        }

        if key in self._args:
            self._args[key] = value
        elif key in input_to_options:
            self._options[input_to_options[key]] = value
        elif key in verbose:
            self._verbose[verbose[key]] = value
        else:
            raise Exception("The argument '{}' to the perf_analyzer "
                             "is not supported by the model analyzer.".format(key))

class PerfAnalyzer:
    """A wrapper class for the perf_analyzer. This
    class provides an interface for running workloads 
    with perf_analyzer
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
        """Parses the parameters to sweep over and runs
           the perf_analyzer once per configuration.

        Parameters
        ----------
        sweep_params : dictionary of lists
            keys are arguments to perf_analyzer, values are 
            list of argument values to run perf_analyzer with.
        
        Returns
        -------
        List of tuples of (params, output) where params are the 
        set of parameters provided for that run and output is the 
        stdout from the perf_analyzer.
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
            run_output = self._run_perf_analyzer()

            run_outputs.append((run_config, run_output))
        
        return run_outputs

    def _run_perf_analyzer(self):
        """Runs the perf_analyzer tool locally 
            and synchronously
        """
        cmd = ['perf_analyzer']
        cmd += self._config.to_cli_string().replace('=', ' ').split()

        # Synchronously start and finish run

        try:    
            out = check_output(cmd, stderr=STDOUT, encoding='utf-8')
        except CalledProcessError as e:
            raise Exception("perf analyzer returned with exit status {} : {}"
                                        .format(e.returncode, e.output))

        return out