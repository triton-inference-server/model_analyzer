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

import unittest

from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.triton.server.server_local_factory import TritonServerLocalFactory
from model_analyzer.analyzer.perf_analyzer.perf_analyzer import PerfAnalyzer
from model_analyzer.analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.analyzer.perf_analyzer.perf_record import PerfRecord
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

# Test Parameters
MODEL_LOCAL_PATH = '/model_analyzer/models'
MODEL_REPOSITORY_PATH = '/model_analyzer/models'
TRITON_VERSION = '20.09'
TEST_MODEL_NAME = 'classification_chestxray_v1'
CONFIG_TEST_ARG = 'sync'
TEST_RUN_PARAMS = {
    'batch-size': [1, 2],
    'concurrency-range': [2, 4]
}
PERF_RECORD_EXAMPLE = (
    "*** Measurement Settings ***\n"
    "  Batch size: 1\n"
    "  Measurement window: 5000 msec\n\n"
    "Request concurrency: 4\n"
    "  Client:\n"
    "    Request count: 100\n"
    "    Throughput: 40.8 infer/sec\n"
    "    Avg latency: 2000 usec\n"
    "    p50 latency: 2000 usec\n"
    "    p90 latency: 2000 usec\n"
    "    p95 latency: 2000 usec\n"
    "    p99 latency: 2000 usec\n"
    "  Server:\n"
    "    Inference count: 100\n"
    "    Execution count: 100\n"
    "    Successful request count: 100\n"
    "    Avg request latency: 2000 usec\n\n"
    "Inferences/Second vs. Client Average Batch Latency\n"
    "Concurrency: 1, throughput: 45 infer/sec, latency 22222 usec\n")


class TestPerfAnalyzerMethods(unittest.TestCase):

    def setUp(self):
        # PerfAnalyzer config for all tests
        self.config = PerfAnalyzerConfig()
        self.config['model-name'] = TEST_MODEL_NAME

        # Triton Server
        self.server = None

    def test_perf_analyzer_config(self):
        # Check config initializations
        self.assertIsNone(self.config[CONFIG_TEST_ARG],
                          msg="Server config had unexpected initial"
                          f" value for {CONFIG_TEST_ARG}")
        # Set value
        self.config[CONFIG_TEST_ARG] = True

        # Test get again
        self.assertTrue(self.config[CONFIG_TEST_ARG],
                        msg=f"{CONFIG_TEST_ARG} was not set")

        # Try to set an unsupported config argument, expect failure
        with self.assertRaises(TritonModelAnalyzerException,
                               msg="Expected exception on trying to set"
                               "unsupported argument in perf_analyzer"
                               "config"):
            self.config['dummy'] = 1

    def test_run(self):
        # Now create a server config
        server_config = TritonServerConfig()
        server_config['model-repository'] = MODEL_REPOSITORY_PATH

        # Create server, PerfAnalyzer, and wait for server ready
        factory = TritonServerLocalFactory()
        self.server = factory.create_server(
            model_path=MODEL_LOCAL_PATH,
            version=TRITON_VERSION,
            config=server_config)
        client = PerfAnalyzer(config=self.config)

        self.server.start()
        self.server.wait_for_ready(num_retries=10)

        # run job with test sweep params
        outputs = client.run_job(sweep_params=TEST_RUN_PARAMS)

        # Ensure correct number of runs
        self.assertEqual(len(outputs), 4)

        self.server.stop()

    def test_perf_record(self):
        # Create a perf record from the example
        record = PerfRecord(PERF_RECORD_EXAMPLE)

        # Now check that the output was correctly parsed
        self.assertEqual(str(record), PERF_RECORD_EXAMPLE.rsplit('\n', 3)[0])

    def tearDown(self):
        # In case test raises exception
        if self.server is not None:
            self.server.stop()


if __name__ == '__main__':
    unittest.main()
