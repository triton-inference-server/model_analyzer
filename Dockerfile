# Copyright 2020, NVIDIA CORPORATION.
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

FROM nvcr.io/nvidia/tensorrtserver:20.02-py3-clientsdk AS base
#Install Docker for starting server in standalone Docker container
RUN apt-get clean && \
  apt-get update && \
  apt install docker.io -y

FROM mcr.microsoft.com/dotnet/core/sdk:3.1-buster AS build
WORKDIR /src
COPY src .
RUN dotnet restore "model-analyzer.csproj"
RUN dotnet publish "model-analyzer.csproj" -c Release -o /app/publish

FROM base
WORKDIR /app
COPY --from=build /app/publish .
ENTRYPOINT ["./model-analyzer"]