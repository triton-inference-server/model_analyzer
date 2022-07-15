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

import os
import csv
import pandas as pd


class ExperimentPlot:
    """
    A wrapper class around Pandas and Matplotlib to load and generate
    plots from the csv file and save the plotted figures.

    >>> plotter = ExperimentPlot(filepath=...)
    >>> plotter.add_metric_scores(metric_fn=..., name="score")

    >>> plotter.data
    >>>     A  B  C  score
        0  10  8 13    2.0
        1   1  4 21    3.0
        2   5  6  3    1.0
        3   3 17 11    1.5

    >>> top_2_best = plotter.get_top_n_values(metric_name="score",
                                              n=2,
                                              ascending=False)
    >>> top_2_best
    >>>     A  B  C  score
        0   1  4 21    3.0
        1  10  8 13    2.0

    >>> top_2_worst = plotter.get_top_n_values(metric_name="score",
                                               n=2,
                                               ascending=True)
    >>> top_2_worst
    >>>     A  B  C  score
        0   5  6  3    1.0
        1   3 17 11    1.5

    >>> plot_data = {
            "Top-2 Best": top_2_best["C"],
            "Top-2 Worst": top_2_worst["C"]
        }
    >>> plotter.plot_histogram(plot_data,
                               title="Top-2 Best/Worst Histogram of C",
                               x_label="C")
    >>> plotter.save(filepath=..., dpi=300)
    """

    def __init__(self, filepath):
        self._ax = None
        self._data = self._load_data(filepath)

    def _load_data(self, filepath):
        """
        Loads the data saved in a csv file.
        """
        self._data = pd.read_csv(filepath)

    @property
    def data(self):
        """
        Returns
        -------
        self._data : pd.DataFrame
            A pandas dataframe that contains experiment results.
        """
        return self._data

    def get_top_n_values(self, metric_name, n=10, ascending=False):
        """
        Get top-n results based on the metric specified.

        Parameters
        ----------
        metric_name : str
            A key to sort the results by.
        n : int
            An integer to specify how many results to include.
        ascending : bool
            Whether or not we want the top-n best or worst results.

        Returns
        -------
        result : pd.DataFrame
            A top-n average results based on the metric specified.
        """
        if metric_name not in self._data:
            raise KeyError(f"{metric_name} does not exist in the data.")

        result = self._data.sort_values(by=metric_name, ascending=ascending)
        result = result.head(n=n).reset_index(drop=True)
        return result

    def plot_histogram(self, data, title, x_label, y_label="Frequency",
                       xticks=None, bins=None, align="left", figsize=None,
                       fontsize=None, alpha=0.7):
        """
        Plot histogram of the given data.

        Parameters
        ----------
        data : dict
            A dictionary that maps a name to its corresponding
            data of type pd.DataFrame.
        """
        self._ax = data.plot(kind="hist", xticks=xticks, bins=bins, align=align,
                     figsize=figsize, fontsize=fontsize, alpha=alpha)

        self._ax.set_title(title, fontsize=fontsize)
        self._ax.set_xlabel(x_label, fontsize=fontsize)
        self._ax.set_ylabel(y_label, fontsize=fontsize)
        self._ax.legend(fontsize=fontsize)

    def add_metric_scores(self, metric_fn, name="metric_score"):
        """
        Insert an additional column to the data that measures the score of
        each row or configuration.

        Parameters
        ----------
        metric_fn : Callable
            A custom metric function to measure the goodness.
        name : str
            A name for the computed metric scores. Default to 'metric_score'.
        """
        scores = metric_fn(self._data)
        self._data[name] = scores

    def save(self, filepath, dpi=None):
        """
        Save the plotted figure to a specified path.

        Parameters
        ----------
        filepath : str
            A path to save the plotted figure.
        dpi : int (optional)
            A resolution of the figure to be saved. Default to 100.
        """
        if self._ax:
            self._fig.savefig(filepath, dpi=dpi)
        else:
            raise ValueError("A plot has not been generated yet. "
                             "Please plot the figure before saving it.")
