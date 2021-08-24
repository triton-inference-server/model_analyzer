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

from .report import Report
import base64
import pdfkit


class PDFReport(Report):
    """
    A report that gets
    constructed in html and
    written to disk as a PDF
    """

    def __init__(self):
        self._head = ""
        self._body = ""

    def head(self):
        """
        Get the head section of
        the html document
        """

        return f'<head><style>{self._head}</style></head>'

    def body(self):
        """
        Get the body section of
        the html document
        """

        return f'<body>{self._body}</body>'

    def document(self):
        """
        Get the html content of
        this PDFReport
        """

        return f'<html>{self.head()}{self.body()}</html>'

    def add_title(self, title):
        """
        Parameters
        ----------
        title: str
            The title of the report
        """

        self._body += f'<center><h1>{title}</h1></center>'

    def add_subheading(self, subheading):
        """
        Parameters
        ----------
        subheading: str
            The subheading of the given section
        """

        self._body += f'<h3>{subheading}</h3>'

    def add_images(self,
                   images,
                   image_captions,
                   image_width=100,
                   float="center"):
        """
        Parameters
        ----------
        images: list of str
            The fullpaths to the image to
            be added to this image row
        image_captions : list of str
            List of image captions
        image_width: int 
            Percentage of the the row of images
            will occupy.
        float: str
            Alignment of the div containing each image in the row
        """

        image_row = ""
        for img, caption in zip(images, image_captions):
            with open(img, "rb") as image_file:
                data_uri = base64.b64encode(image_file.read()).decode('ascii')
                image_row += f"<div class=\"image\" style=\"float:{float};width:{image_width//len(images)}%\">"
                image_row += f"<img src=\"data:image/png;base64,{data_uri}\" style=\"width:100%\">"
                image_row += f"<center><div style=\"font-weight:bold;font-size:12;padding-bottom:20px\">{caption}</div></center>"
                image_row += "</div>"

        self._body += f"<center><div>{image_row}</div></center>"

    def add_paragraph(self, paragraph, font_size=14):
        """
        Parameters
        ----------
        paragraph: str
            The text to add to
            the report as a paragraph
        """

        self._body += f'<div style=\"font-size:{font_size}\"><p>{paragraph}</p></div>'

    def add_line_breaks(self, num_breaks=1):
        """
        Parameters
        ----------
        num_breaks: paragraph
            The text to add to
            the report as a paragraph
        """

        for _ in range(num_breaks):
            self._body += '<br>'

    def add_table(self, table):
        """
        Parameters
        ----------
        table: ResultTable
            The table we want to add
        """

        def table_style(border="1px solid black",
                        padding="5px 10px",
                        font_size="11pt",
                        text_align="center",
                        width="80%"):
            return (f"border: {border};"
                    f"border-collapse: collapse;"
                    f"text-align: {text_align};"
                    f"width: {width};"
                    f"padding: {padding};"
                    f"font-size: {font_size}")

        html_table = ""
        # Add headers
        headers = "".join([
            f'<th style=\"{table_style()}\">{h}</th>' for h in table.headers()
        ])
        html_table += f'<tr>{headers}</tr>'

        # Add data
        for i in range(table.size()):
            row_data = "".join([
                f'<td style=\"{table_style()}\">{d}</td>'
                for d in table.get_row_by_index(i)
            ])
            html_table += f'<tr>{row_data}</tr>'

        # Wrap with table details
        html_table = f'<table style=\"{table_style()}\">{html_table}</table>'
        self._body += f'<center>{html_table}</center>'

    def write_report(self, filename):
        """
        Write the report to disk with
        filename

        Parameters
        ----------
        filename : str
            The name of the report
        """

        pdfkit.from_string(self.document(),
                           f'{filename}',
                           options={'quiet': ''})
