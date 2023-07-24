<!--
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Changelog

## 1.30.0 (2023-06-29)

- Implemented periodic checkpointing
- Added support for custom docker args
- Detect and handle invalid metrics url
- Profile will now automatically create the default detailed reports

## 1.29.0 (2023-06-01)

- `request-rate-range` can now be searched in brute mode
- Capture PA errors in a log file
- Added detection for Triton Server launch failures
- Added `cpu_only` option for ensemble composing models
- Added binary concurrency search to quick search mode
- Added binary parameter search to brute search mode

## 1.28.0 (2023-04-27)

- [Support for BLS models added](docs/config_search.md#bls-model-search)

## 1.27.0 (2023-03-29)

- [Support for ensemble models added](docs/config_search.md#ensemble-model-search)
- [Multi-model quick start guide added](docs/mm_quick_start.md)

## 1.21.0 (2022-11-04)

- [Multi-model search mode added](docs/config_search.md#multi-model-search-mode)

## 1.20.0 (2022-10-04)

- [Quick search mode added](docs/config_search.md#quick-search-mode)

## 1.14.0 (2022-03-29)

- Added support to allow the user to specify a max batch size when automatically sweeping

## 1.10.0 (2021-11-19)

- [Added ability to automatically sweep batch size, instance count and concurrency](docs/config_search.md#automatic-brute-search)

## 1.5.0 (2021-06-19)

- Initial release of Model Analyzer
