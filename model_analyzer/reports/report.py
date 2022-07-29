# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from abc import ABC, abstractmethod


class Report(ABC):
    """
    Defines functions that need to 
    be implemented by all report 
    types

    This will be a html
    """

    @abstractmethod
    def add_title(self, title):
        """
        Parameters
        ----------
        title: str
            The title of the report
        """

    @abstractmethod
    def add_subheading(self, subheading):
        """
        Parameters
        ----------
        subheading: str
            The subheading of the given section
        """

    @abstractmethod
    def add_images(self, images, image_captions):
        """
        Parameters
        ----------
        images: list of str
            The fullpaths to the image to
            be added to this image row
        image_captions : list of str
            List of image captions
        """

    @abstractmethod
    def add_paragraph(self, paragraph):
        """
        Parameters
        ----------
        title: paragraph
            The text to add to
            the report as a paragraph
        """

    @abstractmethod
    def write_report(self, filename):
        """
        Write the report to disk with
        filename

        Parameters
        ----------
        filename : str
            The name of the report
        """

    @abstractmethod
    def get_file_extension(self):
        """
        Return the file extension for 
        the type of report
        """
