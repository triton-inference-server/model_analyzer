#!/usr/bin/env python3

# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ast
import re
import unittest
from pathlib import Path

from .common import test_result_collector as trc

_REPORTS = Path(__file__).resolve().parents[1] / "model_analyzer" / "reports"

# Values that originate outside Model Analyzer -- a model repository, a config
# file or a device query -- and are therefore attacker-controlled when the
# models under test came from a third party.
_TAINTED = re.compile(
    r"\b("
    r"model_name|model_config_name|variant_name|report_key"
    r"|gpu_names|max_memories|composing_config_name|platform"
    r")\b"
)

# An opening or closing tag, i.e. the literal is building markup rather than
# plain prose.
_MARKUP = re.compile(r"<\s*/?\s*[A-Za-z]")

# Sinks in html_report.py that escape on the caller's behalf. Anything that
# reaches the report through one of these does not need escaping first.
_ESCAPING_SINKS = (
    "add_title",
    "add_subheading",
    "add_table",
    "add_images",
)


def _is_markup_literal(node):
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and bool(_MARKUP.search(node.value))
    )


def _flatten_add(node):
    """Flatten a chain of string concatenations into its operands."""

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_add(node.left) + _flatten_add(node.right)
    return [node]


def find_unescaped_interpolations(source):
    """
    Report every place a tainted value is interpolated into a string that
    builds its own markup without going through escape().

    Only markup-building strings are checked. Plain prose handed to
    add_title/add_subheading/add_table/add_images is escaped at the sink,
    so escaping it here too would double-encode it.
    """

    tree = ast.parse(source)
    violations = []
    # ``a + b + c`` nests as BinOp(BinOp(a, b), c); walking visits the chain
    # once per level, so inner nodes are skipped to avoid double counting.
    inner_adds = {
        id(side)
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
        for side in (node.left, node.right)
        if isinstance(side, ast.BinOp) and isinstance(side.op, ast.Add)
    }

    def check(expression_node, lineno):
        segment = ast.get_source_segment(source, expression_node) or ""
        if _TAINTED.search(segment) and "escape(" not in segment:
            violations.append((lineno, segment.strip()))

    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            if not any(_is_markup_literal(v) for v in node.values):
                continue
            for value in node.values:
                if isinstance(value, ast.FormattedValue):
                    check(value.value, value.lineno)

        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and _is_markup_literal(node.func.value)
        ):
            # ``"<br>".join([...])`` -- the separator is markup, so every
            # joined element lands in the document as markup too.
            for argument in node.args:
                for element in ast.walk(argument):
                    if isinstance(element, (ast.Name, ast.Attribute, ast.Call)):
                        check(element, element.lineno)
                        break

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if id(node) in inner_adds:
                continue
            operands = _flatten_add(node)
            if not any(_is_markup_literal(operand) for operand in operands):
                continue
            for operand in operands:
                if isinstance(operand, ast.Subscript):
                    # ``lookup[model_name]`` is only tainted if the container
                    # itself holds outside text; the key is not what lands in
                    # the report.
                    check(operand.value, operand.lineno)
                elif isinstance(operand, (ast.Name, ast.Attribute)):
                    check(operand, operand.lineno)

    return violations


class TestReportEscapingInvariant(trc.TestResultCollector):
    """
    Escaping the report sentences one substitution at a time is only safe
    while every substitution is actually wrapped. A sentence added later
    that interpolates a model name straight into its own markup would
    silently reopen NVBug 6833957, so the rule is enforced over the source
    rather than by review.
    """

    def test_no_tainted_value_is_interpolated_into_markup_unescaped(self):
        offenders = []
        for path in sorted(_REPORTS.glob("*.py")):
            for lineno, segment in find_unescaped_interpolations(path.read_text()):
                offenders.append(f"{path.name}:{lineno}: {segment}")

        self.assertEqual(
            offenders,
            [],
            "A model-derived value is interpolated into report markup without "
            "escape(). Wrap it: f'<strong>{escape(name)}</strong>'.\n  "
            + "\n  ".join(offenders),
        )

    def test_escaping_sinks_still_escape(self):
        """
        The companion half: the methods this invariant trusts to escape on
        the caller's behalf must keep doing so.
        """

        source = (_REPORTS / "html_report.py").read_text()
        tree = ast.parse(source)
        checked = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in _ESCAPING_SINKS:
                body = ast.get_source_segment(source, node) or ""
                self.assertIn(
                    "escape(",
                    body,
                    f"{node.name} is relied on to escape its input but no "
                    f"longer calls escape()",
                )
                checked.add(node.name)

        self.assertEqual(checked, set(_ESCAPING_SINKS))

    def test_invariant_detects_an_unescaped_f_string(self):
        """
        The check has to fail on the thing it exists to catch.
        """

        offending = 'sentence = f"<strong>{model_config_name}</strong> is fast"'
        self.assertEqual(len(find_unescaped_interpolations(offending)), 1)

        fixed = 'sentence = f"<strong>{escape(model_config_name)}</strong> is fast"'
        self.assertEqual(find_unescaped_interpolations(fixed), [])

    def test_invariant_detects_an_unescaped_concatenation(self):
        """
        report_manager.py builds some markup with + rather than an f-string,
        so the concatenation form must be caught too.
        """

        offending = 'out += "<strong>" + model_name + "</strong>: "'
        self.assertEqual(len(find_unescaped_interpolations(offending)), 1)

        fixed = 'out += "<strong>" + escape(model_name) + "</strong>: "'
        self.assertEqual(find_unescaped_interpolations(fixed), [])

    def test_invariant_ignores_plain_prose(self):
        """
        Prose handed to an escaping sink must not be flagged, otherwise the
        fix would be to double-encode it.
        """

        prose = 'caption = f"Latency Breakdown for {report_key}"'
        self.assertEqual(find_unescaped_interpolations(prose), [])

    def test_invariant_detects_an_unescaped_join(self):
        """
        ``"<br>".join(...)`` puts every element into the document as markup.
        """

        offending = 'names = "<br>".join([c.variant_name for c in configs])'
        self.assertEqual(len(find_unescaped_interpolations(offending)), 1)

        fixed = 'names = "<br>".join([escape(c.variant_name) for c in configs])'
        self.assertEqual(find_unescaped_interpolations(fixed), [])


if __name__ == "__main__":
    unittest.main()
