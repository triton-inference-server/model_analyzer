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

import sys
from model_analyzer.constants import MAX_NUMBER_OF_INTERRUPTS
from model_analyzer.state.analyzer_state import AnalyzerState
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

import signal
import logging
import pickle
import os
import glob


class AnalyzerStateManager:
    """
    Maintains the state of the Model Analyzer
    """

    def __init__(self, config):
        self._config = config
        self._exiting = 0
        self._checkpoint_dir = config.checkpoint_directory
        self._state_changed = False

        if os.path.exists(self._checkpoint_dir):
            self._checkpoint_index = self._latest_checkpoint() + 1
        else:
            os.makedirs(self._checkpoint_dir)
            self._checkpoint_index = 0
        signal.signal(signal.SIGINT, self.interrupt_handler)

    def starting_fresh_run(self):
        """
        Returns
        -------
        True if starting a fresh run
        False if checkpoint found and loaded
        """

        return self._starting_fresh_run

    def exiting(self):
        """
        Returns
        -------
        True if interrupt handler ran
        even once, False otherwise
        """

        return self._exiting > 0

    def get_state_variable(self, name):
        """
        Get a named variable from
        the current AnalyzerState

        Parameters
        ----------
        name : str
            The name of the variable
        """
        return self._current_state.get(name)

    def set_state_variable(self, name, value):
        """
        Set a named variable from
        the current AnalyzerState

        Parameters
        ----------
        name: str
            The name of the variable
        value: Any 
            the value to set for that variable
        """

        self._state_changed = True
        self._current_state.set(name, value)

    def load_checkpoint(self):
        """
        Load the state of the Model Analyzer from
        most recent checkpoint file, also 
        set whether we are starting a fresh run
        """

        latest_checkpoint_file = os.path.join(
            self._checkpoint_dir, f"{self._latest_checkpoint()}.ckpt")
        if os.path.exists(latest_checkpoint_file):
            logging.info(
                f"Loaded checkpoint from file {latest_checkpoint_file}")
            with open(latest_checkpoint_file, 'rb') as f:
                try:
                    state = pickle.load(f)
                except EOFError:
                    raise TritonModelAnalyzerException(
                        f'Checkpoint file {latest_checkpoint_file} is'
                        ' empty or corrupted. Remove it from checkpoint'
                        ' directory.')
            self._current_state = state
            self._starting_fresh_run = False
        else:
            self._current_state = AnalyzerState()
            self._starting_fresh_run = True
            logging.info("No checkpoint file found, starting a fresh run.")

    def save_checkpoint(self):
        """
        Saves the state of the model analyzer to disk
        if there has been a change since the last checkpoint

        Parameters
        ----------
        state: AnalyzerState
            The state object to be saved
        """

        ckpt_filename = os.path.join(self._checkpoint_dir,
                                     f"{self._checkpoint_index}.ckpt")
        if self._state_changed:
            with open(ckpt_filename, 'wb') as f:
                pickle.dump(self._current_state, f)
            logging.info(f"Saved checkpoint to {ckpt_filename}.")

            self._checkpoint_index += 1
            self._state_changed = False
        else:
            logging.info(
                f"No changes made to analyzer data, no checkpoint saved.")

    def interrupt_handler(self, signal, frame):
        """
        A signal handler to properly
        shutdown the model analyzer on
        interrupt
        """

        self._exiting += 1
        logging.info(
            f'Received SIGINT {self._exiting}/{MAX_NUMBER_OF_INTERRUPTS}. '
            'Will attempt to exit after current measurement.')
        if self._exiting >= MAX_NUMBER_OF_INTERRUPTS:
            logging.info(
                f'Received SIGINT maximum number of times. Saving state and exiting immediately. '
                'perf_analyzer and tritonserver may still be running...')
            self.save_checkpoint()
            sys.exit(1)

    def _latest_checkpoint(self):
        """
        Get the highest index checkpoint file in the
        checkpoint directory, return its index.
        """

        checkpoint_files = glob.glob(
            os.path.join(self._checkpoint_dir, '*.ckpt'))
        if not checkpoint_files:
            return -1
        try:
            return max([
                int(os.path.split(f)[1].split('.')[0]) for f in checkpoint_files
            ])
        except Exception as e:
            raise TritonModelAnalyzerException(e)
