FROM nvcr.io/nvidia/tensorrtserver:20.02-py3-clientsdk AS base
#Install Docker for starting server in standalone Docker container
RUN apt-get update && \
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