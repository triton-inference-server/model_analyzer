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

import requests
import docker
import os
import time

from subprocess import Popen, PIPE, STDOUT, TimeoutExpired

TRITONSERVER_IMAGE = 'nvcr.io/nvidia/tritonserver:'

WAIT_FOR_READY_NUM_RETRIES = 100
LOCAL_HTTP_PORT = 8000
LOCAL_GRPC_PORT = 8001
LOCAL_METRICS_PORT = 8002

SERVER_OUTPUT_TIMEOUT_SECS=5

class TritonServerConfig:
    """
    A config class to set arguments to the Triton Inference
    Server. An argument set to None will use the server default.
    """
    
    def __init__(self):
        # Args will be a dict with the string representation as key
        self._server_args = {
            # Logging
            'log-verbose' : None,
            'log-info' : None,
            'log-warning' : None,
            'log-error' : None,
            'id' : None,
        
            # Model Repository
            'model-store' : None,
            'model-repository' : None,
            
            # Exit
            'exit-timeout-secs' : None,
            'exit-on-error' : None,

            # Strictness
            'strict-model-config' : None,
            'strict-readiness' : None,
            
            # API Servers
            'allow-http' : None,
            'http-port' : None,
            'http-thread-count' : None,
            'allow-grpc' : None,
            'grpc-port' : None,
            'grpc-infer-allocation-pool-size' : None,
            'grpc-use-ssl' : None,
            'grpc-server-cert' : None,
            'grpc-server-key' : None,
            'grpc-root-cert' : None,
            'allow-metrics' : None,
            'allow-gpu-metrics' : None,
            'metrics-port' : None,
            
            # Tracing
            'trace-file' : None,
            'trace-level' : None,
            'trace-rate' : None,
            
            # Model control
            'model-control-mode' : None,
            'repository-poll-secs' : None,
            'load-model' : None,

            # Memory and GPU
            'pinned-memory-pool-byte-size' : None,
            'cuda-memory-pool-byte-size' : None,
            'min-supported-compute-capability' : None,
            
            # Backend config
            'backend-directory' : None,
            'backend-config' : None,
            
            'allow-soft-placement' : None,
            'gpu-memory-fraction' : None,
            'tensorflow-version' : None
        }

    def to_cli_string(self):
        """
        Utility function to convert a config into a
        string of arguments to the server with CLI.
        """
        return ' '.join(['--{}={}'.format(key,val) if val else '' \
                        for key,val in self._server_args.items()])

    def __getitem__(self, key):
        return self._server_args[key]
    
    def __setitem__(self, key, value):
        if key in self._server_args:
            self._server_args[key] = value
        else:
            raise Exception("The argument '{}' to the Triton Inference Server"
                             "is not supported by the model analyzer.".format(key))
    
class TritonServerFactory:
    """
    A factory for creating TritonServer instances
    """

    def __init__(self):
        pass

    def create_server(self, run_type, model_path, version, config):
        if run_type is 'docker':
            return TritonServerDocker(version=version, model_path=model_path, config=config)
        elif run_type is 'local':
            return TritonServerLocal(version=version, model_path=model_path, config=config)
        else:
            raise Exception("The run environment '{}' for Triton Inference Server"
                            "is not supported by the model analyzer.".format(run_type))

class TritonServer:
    """
    Defines the interface for the objects created by 
    TritonServerFactory
    """
    def __init__(self, version, model_path, config):
        self._version = version
        self._server_config = config
        self._model_path = model_path
        assert self._server_config['model-repository'], \
            "Triton Server requires --model-repository argument to be set."    

        self._logs = None

    def start(self):
        """
        Starts the tritonserver 
        """
        raise NotImplementedError()
    
    def stop(self):
        """
        Stops and cleans up after the server
        """
        raise NotImplementedError()

    def wait_for_ready(self, num_retries=WAIT_FOR_READY_NUM_RETRIES):

        # FIXME this should not really be a server function
        if self._server_config['allow-http'] is not False:
            http_port = self._server_config['http-port'] or 8000
            url = "http://localhost:{}/v2/health/ready".format(http_port)
        else:
            # FIXME to use GRPC to check for ready also
            raise Exception('allow-http must be True in order to use wait_for_server_ready')
        
        retries = num_retries
        # poll ready endpoint for number of retries
        while num_retries > 0:
            try:
                r = requests.get(url)
                if r.status_code == 200:
                    return True
            except requests.exceptions.RequestException as e:
                pass
            time.sleep(0.1)
            retries -= 1
        
        # If num_retries is exceeded return an exception
        raise Exception("Server not ready : num_retries : {}".format(num_retries))

class TritonServerDocker(TritonServer):
    """
    Concrete Implementation of TritonServer interface that runs
    triton in a docker container.
    """
    def __init__(self, version, model_path, config):
        super().__init__(version=version, model_path=model_path, config=config)

        self._docker_client = docker.from_env()
        self._tritonserver_image = TRITONSERVER_IMAGE + version + '-py3'
        self._tritonserver_container = None
        

    def start(self):
        """
        Starts the tritonserver docker container
        """
        # get devices using CUDA_VISIBLE_DEVICES
        CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '') # default to cpu

        if CUDA_VISIBLE_DEVICES != '':
            device_ids = CUDA_VISIBLE_DEVICES.split(',')
            devices = [
                docker.types.DeviceRequest(
                    device_ids=device_ids, 
                    capabilities=[['gpu']]
                    )
                ]
        else:
            devices = None

        # Mount required directories
        volumes = {
            self._model_path : { 
                'bind' : self._server_config['model-repository'],
                'mode' : 'rw'
                }   
        }

        # Map ports, use config values but set to server defaults if not specified
        server_http_port = self._server_config['http-port'] or 8000
        server_grpc_port = self._server_config['grpc-port'] or 8001
        server_metrics_port = self._server_config['metrics-port'] or 8002
        
        ports = {
            server_http_port : LOCAL_HTTP_PORT,
            server_grpc_port : LOCAL_GRPC_PORT,
            server_metrics_port : LOCAL_METRICS_PORT
        }

        # Run the docker container
        self._tritonserver_container = self._docker_client.containers.run(
                                            image=self._tritonserver_image,
                                            device_requests=devices,
                                            volumes=volumes,
                                            ports=ports,
                                            publish_all_ports=True,
                                            tty=True,
                                            stdin_open=True,
                                            detach=True)
                                        
        # Run the command in the container
        cmd = '/opt/tritonserver/bin/tritonserver ' + \
            self._server_config.to_cli_string()
        
        self._tritonserver_log = \
            self._tritonserver_container.exec_run(cmd, stream=True)

    def stop(self):
        """
        Stops the tritonserver docker container
        and cleans up docker client
        """
        if self._tritonserver_container is not None:
            self._tritonserver_container.stop()
            self._tritonserver_container.remove()
            
            self._tritonserver_container = None
            self._docker_client.close()

class TritonServerLocal(TritonServer):
    """
    Concrete Implementation of TritonServer interface that runs
    tritonserver locally as as subprocess.
    """
    def __init__(self, version, model_path, config):
        super().__init__(version=version, model_path=model_path, config=config)
        self._tritonserver_process = None

    def start(self):
        """
        Starts the tritonserver  container locally
        """
        # Run the subprocess
        cmd = ['/opt/tritonserver/bin/tritonserver']
        cmd += self._server_config.to_cli_string().replace('=', ' ').split()
        
        self._tritonserver_process = \
             Popen(cmd, stdout=PIPE, stderr=STDOUT, universal_newlines=True)

    def stop(self):
        """
        Stops the running tritonserver
        """
        if self._tritonserver_process is not None:
            self._tritonserver_process.terminate()
            try:
                output, _ = self._tritonserver_process.communicate(
                    timeout=SERVER_OUTPUT_TIMEOUT_SECS)
            except TimeoutExpired:
                self._tritonserver_process.kill()
                output, _ = self._tritonserver_process.communicate()
            self._tritonserver_process = None

