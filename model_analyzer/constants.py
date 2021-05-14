# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

# Config constants
CONFIG_PARSER_SUCCESS = 1
CONFIG_PARSER_FAILURE = 0

# Result Table constants
RESULT_TABLE_COLUMN_PADDING = 2

# Result Comparator Constants
COMPARISON_SCORE_THRESHOLD = 0.005

# Run Search
THROUGHPUT_GAIN = 0.05

# Reports
TOP_MODELS_REPORT_KEY = "Best Configs Across All Models"

# State Management
MAX_NUMBER_OF_INTERRUPTS = 3

# Perf Analyzer
MAX_INTERVAL_CHANGES = 10
MEASUREMENT_WINDOW_STEP = 1000
MEASUREMENT_REQUEST_COUNT_STEP = 50
INTERVAL_SLEEP_TIME = 1
PERF_ANALYZER_MEASUREMENT_WINDOW = 5000
PERF_ANALYZER_MEASUREMENT_REQUEST_COUNT = 50

# Triton Server
SERVER_OUTPUT_TIMEOUT_SECS = 5
