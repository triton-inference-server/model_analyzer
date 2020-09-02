/*****************************************************************************
Copyright 2020, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*****************************************************************************/

using ModelAnalyzer.Client;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Threading;
using System.Threading.Tasks;

namespace ModelAnalyzer
{
    /// <summary>
    /// Class used to store configuration of Model Analyzer
    /// </summary>

    class ModelAnalyzerConfig
    {
        /// <summary>
        /// List of batch sizes to test
        /// </summary>
        public IList<int> BatchSize { get; set; }

        /// <summary>
        /// List of concurrency values to test
        /// </summary>
        public IList<int> ConcurrencyValues { get; set; }

        /// <summary>
        /// Folder where Triton model is stored
        /// </summary>
        public string ModelFolder { get; set; }

        /// <summary>
        /// Name of model being analyzed
        /// </summary>
        public string ModelName { get; set; }

        /// <summary>
        /// Version of Triton Server and Client
        /// </summary>
        public string TritonVersion { get; set; } = "20.02-py3";

        /// <summary>
        /// Maximum seconds for any single retry attempt
        /// </summary>
        public TimeSpan MaxRetryTime { get; set; } = TimeSpan.FromSeconds(15);

        /// <summary>
        /// Seconds after perf client called to wait before gathering metrics
        /// </summary>
        public TimeSpan PerfClientBufferTime { get; set; } = TimeSpan.FromSeconds(2);

        /// <summary>
        /// Milliseconds to wait between each metric collection
        /// </summary>
        public TimeSpan UpdateFrequency { get; set; } = TimeSpan.FromMilliseconds(100);
    }

    /// <summary>
    /// Class used for analyzing a model and exporting its compute requirements
    /// </summary>

    class ModelAnalyzer
    {
        private readonly IList<int> _BatchSize;
        private readonly IList<int> _ConcurrencyValues;
        private readonly string _ModelFolder;
        private readonly string _ModelName;
        private readonly string _TritonVersion;
        private readonly TimeSpan _MaxRetryTime;
        private readonly TimeSpan _PerfClientBufferTime;
        private readonly TimeSpan _UpdateFrequency;

        /// <summary>
        /// Ports for Triton to assign
        /// </summary>
        private const int HttpPort = 8000, GrpcPort = 8001, MetricsPort = 8002;

        /// <summary>
        /// Seconds to wait before first retry when error
        /// </summary>
        private readonly TimeSpan _InitialRetryDelay = TimeSpan.FromSeconds(1);

        /// <summary>
        /// Loopback IP address for talking to local network
        /// </summary>
        private static readonly IPAddress _IpAddress = IPAddress.Loopback;

        /// <summary>
        /// Maximum seconds between any two retries
        /// </summary>
        private readonly TimeSpan _MaxRetryDelay = TimeSpan.FromSeconds(30);

        /// <summary>
        /// MetricsCollector object for the model metrics
        /// </summary>
        private readonly MetricsCollector _MetricsCollectorModel;

        /// <summary>
        /// MetricsCollector object for the server-only metrics
        /// </summary>
        private readonly MetricsCollector _MetricsCollectorServerOnly;

        /// <summary>
        /// Multiplier for how many more seconds to wait for each successive retry attempt
        /// </summary>
        private const int RetryDelayMultiplier = 2;

        /// <summary>
        /// Name for server container
        /// </summary>
        private const string ServerName = "triton-server";

        /// <summary>
        /// Constructor for ModelAnalyzer
        /// Used for gathering and exporting metrics to screen
        /// </summary>
        /// <param name="config">Configuration object for Model Analyzer</param>
        /// <returns>Model Analyzer instance</returns>
        public ModelAnalyzer(ModelAnalyzerConfig config, MetricsCollector metricsCollectorServerOnly, MetricsCollector metricsCollectorModel)
        {
            _BatchSize = config.BatchSize;
            _ConcurrencyValues = config.ConcurrencyValues;
            _MetricsCollectorModel = metricsCollectorModel;
            _MetricsCollectorServerOnly = metricsCollectorServerOnly;
            _ModelFolder = config.ModelFolder;
            _ModelName = config.ModelName;
            _TritonVersion = config.TritonVersion;
            _MaxRetryTime = config.MaxRetryTime;
            _PerfClientBufferTime = config.PerfClientBufferTime;
            _UpdateFrequency = config.UpdateFrequency;
        }

        /// <summary>
        /// Entrypoint for running Model Analyzer locally as a Docker container
        /// </summary>
        public void RunLocal()
        {
            using var clientProcess = new Process();
            using var serverProcess = new Process();

            try
            {
                StartServer(serverProcess);
                Console.CancelKeyPress += delegate
                {
                    StopContainer(ServerName);
                };

                var clientInfo = new ProcessStartInfo("docker", $"run --rm --name triton-client " +
                    $"--net=host nvcr.io/nvidia/tensorrtserver:{_TritonVersion}-clientsdk perf_client " +
                    $"-m {_ModelName} ");
                clientProcess.StartInfo = clientInfo;

                Run(clientProcess);

                Console.CancelKeyPress -= delegate
                {
                    StopContainer(ServerName);
                };
            }
            finally
            {
                StopContainer(ServerName);
            }
        }

        /// <summary>
        /// Entrypoint for running Model Analyzer in a Kubernetes pod,
        /// alongside a Triton server container
        /// </summary>
        public void RunK8s()
        {
            using var clientProcess = new Process();

            var clientProcessInfo = new ProcessStartInfo("perf_client", $"-m {_ModelName} " +
                $"");

            clientProcess.StartInfo = clientProcessInfo;

            Run(clientProcess);
        }

        /// <summary>
        /// High-level function running Model Analyzer
        /// </summary>
        /// <param name="clientProcess">Client process to run, excluding batch and concurrency values.</param>
        public void Run(Process clientProcess)
        {

            HideProcessInput(clientProcess);

            // Steps:
            // 1. Verify that Triton is running.
            // 2. Gather server-only metrics, if they have not yet been gathered.
            // 3. Use Triton API to load model into server.
            // 4. Use Perf Client to load model into memory.
            // 5. For each concurrency and batch size configuration, run the model in Perf Client.
            // 6. As each configuration is running, gather metrics..
            // 7. Triton server and client's associated processes and containers are cleaned up.

            WaitUntilPortReady();

            if (_MetricsCollectorServerOnly._MetricsSamples.Count == 0)
            {
                _MetricsCollectorServerOnly.AddNewSample(ServerName, 0, 0);
                _MetricsCollectorServerOnly.UpdateLatestMetricsThroughput("0 infer/sec");
                _MetricsCollectorServerOnly.GenerateMetrics();
            }

            LoadModel();

            var baseArgs = clientProcess.StartInfo.Arguments;

            //Use Perf Client to load model into memory.
            HideProcessOutput(clientProcess);
            RunModel(clientProcess);
            clientProcess.WaitForExit();

            foreach (var batch in _BatchSize)
            {
                if (batch < 0)
                {
                    Console.WriteLine($"Invalid batch size: {batch}: skipping");
                    continue;
                }

                foreach (var concurrency in _ConcurrencyValues)
                {
                    if (concurrency < 0)
                    {
                        Console.WriteLine($"Invalid concurrency value: {concurrency}: skipping");
                        continue;
                    }


                    _MetricsCollectorModel.AddNewSample(_ModelName, batch, concurrency);

                    try
                    {
                        clientProcess.StartInfo.Arguments = string.Concat(baseArgs,
                        $"--percentile=96 -b {batch} --concurrency-range {concurrency}:{concurrency}");
                        Console.WriteLine($"\n{_ModelName}");
                        RunModel(clientProcess);

                        Task.Run(
                            () => ParseThroughput(clientProcess, _MetricsCollectorModel));

                        var parseError = new Task(
                            () => ParseError(clientProcess));
                        parseError.Start();

                        GetMetricsRunningModel(clientProcess, _MetricsCollectorModel);

                        parseError.Wait();

                        KillProcess(clientProcess);
                    }
                    catch (Exception exception)
                    {
                        _MetricsCollectorModel.DeletesLatestSample();
                        Console.WriteLine(exception.Message);
                    }
                }
            }
        }

        /// <summary>
        /// Checks that Triton server is running
        /// </summary>
        /// <returns>True if server running, else false</returns>
        public static bool IsServerRunning()
        {
            var client = new TritonGrpcClient();
            if (IsPortReady(client))
                return true;

            return false;
        }

        /// <summary>
        /// Starts Triton server
        /// </summary>
        /// <param name="serverProcess">Server process to start.</param>
        /// <returns>Returns started process.</returns>
        public Process StartServer(Process serverProcess)
        {
            var pullProcess = new Process()
            {
                StartInfo = new ProcessStartInfo("docker", $"pull nvcr.io/nvidia/tensorrtserver:{ _TritonVersion} "),
            };
            HideProcessOutput(pullProcess);
            pullProcess.Start();
            pullProcess.WaitForExit();

            var serverProcessInfo = new ProcessStartInfo("docker", $"run --gpus=1 --rm --name {ServerName} " +
                $"-p{HttpPort}:{HttpPort} -p{GrpcPort}:{GrpcPort} -p{MetricsPort}:{MetricsPort} " +
                $"-v{_ModelFolder}:/models nvcr.io/nvidia/tensorrtserver:{_TritonVersion} " +
                $"./bin/trtserver --model-repository=/models --model-control-mode=explicit");
            serverProcess.StartInfo = serverProcessInfo;
            HideProcessInput(serverProcess);
            HideProcessOutput(serverProcess);

            serverProcess.Start();

            return serverProcess;
        }

        /// <summary>
        /// Hides input of process
        /// </summary>
        /// <param name="process">Process whose input to hide.</param>
        private void HideProcessInput(Process process)
        {
            process.StartInfo.CreateNoWindow = true;
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.RedirectStandardInput = true;
        }

        /// <summary>
        /// Hides output of process
        /// </summary>
        /// <param name="process">Process whose output to hide.</param>
        private void HideProcessOutput(Process process)
        {
            process.StartInfo.RedirectStandardError = true;
            process.StartInfo.RedirectStandardOutput = true;
        }

        /// <summary>
        /// Wait on process until the server's GRPC port is ready
        /// </summary>
        /// <param name="process">Process whose output to hide.</param>
        public void WaitUntilPortReady()
        {
            var client = new TritonGrpcClient();
            var currentRetryDelay = _InitialRetryDelay;
            var currentRetryTime = TimeSpan.Zero;

            while (!IsPortReady(client))
            {
                currentRetryDelay = currentRetryDelay > _MaxRetryDelay ? _MaxRetryDelay : currentRetryDelay;
                Thread.Sleep(currentRetryDelay);
                currentRetryTime += currentRetryDelay;
                currentRetryDelay *= RetryDelayMultiplier;
                if (currentRetryTime >= _MaxRetryTime)
                {
                    throw new TimeoutException("Unable to connect to Triton port: maximum attempts reached");
                }
            }
        }

        /// <summary>
        /// Checks if the server's GRPC port is ready
        /// </summary>
        /// <param name="client">Triton's GRPC client.</param>
        /// <returns>True if port ready, else false.</returns>
        public static bool IsPortReady(TritonGrpcClient client)
        {
            bool ready;
            try
            {
                ready = client.GetStatus(_IpAddress, GrpcPort).ReadyState.Equals(ServerReadyState.ServerReady);
            }
            catch (Grpc.Core.RpcException exception) when (exception.StatusCode == Grpc.Core.StatusCode.Unavailable)
            {
                return false;
            }
            return ready;
        }

        /// <summary>
        /// Loads model on server
        /// </summary>
        public void LoadModel()
        {
            var grpcClient = new TritonGrpcClient();
            var models = new List<string> { _ModelName };
            grpcClient.LoadModels(models, _IpAddress, GrpcPort);
            WaitUntilModelsLoaded(grpcClient);
        }

        /// <summary>
        /// Process waits until the model is ready or times out
        /// </summary>
        /// <param name="client">Triton's GRPC client.</param>
        private void WaitUntilModelsLoaded(TritonGrpcClient client)
        {
            var timeoutSeconds = (int)_MaxRetryTime.TotalSeconds;

            for (var i = 0; i < timeoutSeconds; i++)
            {
                if (IsModelReady(client)) return;

                Thread.Sleep(TimeSpan.FromSeconds(1));
            }

            throw new TimeoutException("Maximum loading time exceeded: models not yet ready");
        }

        /// <summary>
        /// Checks if model is loaded on server
        /// </summary>
        /// <param name="client">Triton's GRPC client.</param>
        /// <returns>Returns true if model loaded, else false.</returns>
        private bool IsModelReady(TritonGrpcClient client)
        {
            var models = client.GetStatus(_IpAddress, GrpcPort).ModelStatus;
            foreach (var model in models)
            {
                if (model.Key.Equals(_ModelName))
                {
                    foreach (var status in model.Value)
                        if (!status.Value.Equals(ModelReadyState.ModelReady)) return false;

                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Runs the model via the process
        /// </summary>
        /// <param name="clientProcess">Process to run perf client with model.</param>
        public void RunModel(Process clientProcess)
        {
            clientProcess.Start();
            Thread.Sleep(_PerfClientBufferTime);
        }

        /// <summary>
        /// Parses throughput from standard output
        /// </summary>
        /// <param name="clientProcess">Process to parse throughput from.</param>
        /// /// <param name="collector">MetricsCollector to update throughput for.</param>
        public void ParseThroughput(Process clientProcess, MetricsCollector collector)
        {
            while (!clientProcess.StandardOutput.EndOfStream)
            {
                var line = clientProcess.StandardOutput.ReadLine();
                if (line.Contains("Throughput"))
                {
                    var throughput = line.Split(":").Last().Trim();
                    collector.UpdateLatestMetricsThroughput(throughput);
                }
                Console.WriteLine(line);
            }
        }

        /// <summary>
        /// Throws an error, if parses error from standard error
        /// </summary>
        /// <param name="clientProcess">Process to parse throughput from.</param>
        public void ParseError(Process clientProcess)
        {
            while (!clientProcess.StandardError.EndOfStream)
            {
                var line = clientProcess.StandardError.ReadLine();
                Console.WriteLine(line);
                if (line.Contains("INTERNAL"))
                {
                    throw new Exception("Error running model with specified configuration: skipping");
                }
            }
        }

        /// <summary>
        /// Gathers metrics for running model
        /// </summary>
        /// <param name="clientProcess">Process to run perf client with model.</param>
        public void GetMetricsRunningModel(Process clientProcess, MetricsCollector collector)
        {
            collector.ClearCachedMetrics();

            while (IsRunning(clientProcess))
            {
                collector.UpdateLatestMetrics();
                Thread.Sleep(_UpdateFrequency);
            }
        }

        /// <summary>
        /// Checks if a given process is running
        /// </summary>
        /// <param name="process">Process to check.</param>
        /// <returns>Returns true if process is running, else false.</returns>
        private bool IsRunning(Process process)
        {
            return !(process.WaitForExit(0));
        }

        /// <summary>
        /// Kills a given process, if running
        /// </summary>
        /// <param name="process">Process to kill.</param>
        private void KillProcess(Process process)
        {
            if (!process.HasExited)
            {
                process.Kill();
            }
        }

        /// <summary>
        /// Stops a Docker container
        /// </summary>
        /// <param name="containerName">Name of container to stop.</param>
        private void StopContainer(string containerName)
        {
            var stopProcessInfo = new ProcessStartInfo("docker", $"stop {containerName}");
            using var stopProcess = new Process
            {
                StartInfo = stopProcessInfo
            };
            HideProcessInput(stopProcess);
            HideProcessOutput(stopProcess);
            stopProcess.Start();
            stopProcess.WaitForExit();
        }
    }
}
