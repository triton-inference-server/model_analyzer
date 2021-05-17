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


class ConfigPlot:
    """
    A class representing the configuration used for
    a single plot.
    """

    def __init__(self,
                 name,
                 title=None,
                 x_axis=None,
                 y_axis=None,
                 monotonic=False):
        """
        Parameters
        ----------
        name : str
            Name to use to identify the plot
        title : str
            title of the plot
        x_axis : str
            The metric tag for the x-axis of this plot
        y_axis : str
            The metric tag for the y-axis of this plot
        monotonic: bool
            Whether or not to prune decreasing points in this
            plot
        """

        self._name = name
        self._title = title
        self._x_axis = x_axis
        self._y_axis = y_axis
        self._monotonic = monotonic

    def name(self):
        """
        Returns
        -------
        str
            Name of plot (to which it will be saved)
        """

        return self._name

    def title(self):
        """
        Returns
        -------
        str
            title of the plot
        """

        return self._title

    def x_axis(self):
        """
        Returns
        -------
        str
            tag for x_axis of the plot
        """

        return self._x_axis

    def y_axis(self):
        """
        Returns
        -------
        str
            tag for y axis of the plot
        """

        return self._y_axis

    def monotonic(self):
        """
        Returns
        -------
        bool
            Whether or not to prune
            decreasing points 
        """

        return self._monotonic

    def set_name(self, name):
        """
        Parameters
        -------
        name: str
            Name of plot (to which it will be saved)
        """

        self._name = name

    def set_title(self, title):
        """
        Parameters
        -------
        str
            title of the plot
        """

        self._title = title

    def set_x_axis(self, x_axis):
        """
        Parameters
        -------
        x_axis: str
            tag for x_axis of the plot
        """

        self._x_axis = x_axis

    def set_y_axis(self, y_axis):
        """
        Parameters
        -------
        y_axis: str
            tag for x_axis of the plot
        """

        self._y_axis = y_axis

    def set_monotonic(self, monotonic):
        """
        Parameters
        -------
        monotonic: str
            Whether or not to prune
            decreasing points 
        """

        self._monotonic = monotonic

    @staticmethod
    def from_list(plots):
        """
        Converts a List of plot specs to ConfigPlot.

        Parameters
        ----------
        plots : list
            A list of ConfigPlots
            
        Returns
        -------
        list
            A list of ConfigPlot objects.
        """

        plot_list = []
        for plot in plots:
            plot_list.append(plot.value()[0])
        return plot_list

    @staticmethod
    def from_object(plots):
        """
        Converts a plot spec object to ConfigPlot.

        Parameters
        ----------
        plots : dict
            containing plot names and ConfigObjects
        Returns
        -------
        ConfigPlot
        """

        plot_list = []
        for plot_name, plot_spec in plots.items():
            plot_list.append(ConfigPlot(plot_name, **plot_spec.value()))
        return plot_list

    def __repr__(self):
        plot_object = {
            'name': self._name,
            'title': self._title,
            'x_axis': self._x_axis,
            'y_axis': self._y_axis,
            'monotonic': self._monotonic
        }

        return str(plot_object)
