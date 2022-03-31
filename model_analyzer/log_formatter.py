# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import sys
from model_analyzer.constants import LOGGER_NAME


class LogFormatter(logging.Formatter):
    """ Class to handle formatting of the logger outputs """

    def __init__(self):
        logger = logging.getLogger(LOGGER_NAME)
        self._log_level = logger.getEffectiveLevel()
        super().__init__()
        self.datefmt = "%H:%M:%S"

    def format(self, record):
        front = "%(asctime)s " if self._log_level is logging.DEBUG else ""
        if record.levelno == logging.INFO:
            self._style._fmt = f"{front}[Model Analyzer] %(message)s"
        else:
            self._style._fmt = f"{front}[Model Analyzer] %(levelname)s: %(message)s"
        return super().format(record)


def setup_logging(quiet, verbose):
    """
    Setup logger format

    Parameters
    ----------
    quiet : bool
        If true, don't print anything other than errors
    verbose : bool
        If true and quiet is not true, print debug information
    """

    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level=log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(LogFormatter())
    logger.addHandler(handler)
    logger.propagate = False
