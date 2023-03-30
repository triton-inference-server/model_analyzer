# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack


class TritonPythonModel:

    def initialize(self, args):
        """
        This function initializes pre-trained ResNet50 model.
        """
        self.device = 'cuda' if args["model_instance_kind"] == "GPU" else 'cpu'
        self.model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")\
                        .to(self.device)\
                        .eval()

    def execute(self, requests):
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            result = self.model(
                torch.as_tensor(input_tensor.as_numpy(), device=self.device))
            out_tensor = pb_utils.Tensor.from_dlpack("OUTPUT0",
                                                     to_dlpack(result))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
