import os
import shutil
import base64
import re
import warnings
from unittest import TestCase
from copy import deepcopy
from tempfile import NamedTemporaryFile

import pytest
from testbook import testbook
from matplotlib.testing.compare import compare_images

dir = "examples"
if not os.path.exists(dir):
    dir = os.path.join("..", dir)


class NotebookTester:
    def __init__(self, nb):
        self.nb = nb

    def execute(self, *, check_outputs=False):
        with testbook(self.nb) as tb:
            for i, cell in enumerate(tb.cells):
                tags = cell.get("metadata", {}).get("tags", [])
                ref_cell = deepcopy(cell)

                # execute
                cell = tb.execute_cell(i)

                if check_outputs:
                    try:
                        self.compare_cell_outputs(ref_cell, cell)
                    except:
                        print(
                            f"\nError in {self.nb} cell [{ref_cell.get('execution_count')}] (ID:{ref_cell.get('id')}): Output not reproduced."
                        )
                        print(
                            "Cell source:\n    "
                            + "\n    ".join(ref_cell.get("source", "").split("\n"))
                        )
                        raise

                # yield i, tags, ref_cell, cell, tb

    @classmethod
    def compare_cell_outputs(cls, reference_cell, actual_cell):
        """Compare cell outputs unless remove-tags exist"""

        # ignore cells with remove-output tag
        tags = reference_cell.get("metadata", {}).get("tags", [])
        if "remove-output" in tags or "skip-test" in tags:
            return True  # skip completely

        # ignore stderr/stdout cell outputs if cell tagged with remove-stderr or remove-stdout
        def keep(out):
            if out.get("output_type") == "stream" and f"remove-{out.get('name')}" in tags:
                warnings.warn(
                    f"Not checking {out.get('name')} output of cell [{reference_cell.get('execution_count')}]"
                    f" tagged with remove-{out.get('name')}."
                )
                return False
            return True

        # Make sure these are lists, since filter object can be iterated over only once!
        reference_outputs = list(filter(keep, reference_cell.get("outputs", [])))
        actual_outputs = list(filter(keep, actual_cell.get("outputs", [])))

        try:
            # compare the filtered outputs
            for i, ref_out in enumerate(reference_outputs):
                output_type = ref_out.get("output_type")

                try:
                    act_out = actual_outputs[i]
                except IndexError:
                    raise ValueError(
                        f"Expected {len(reference_outputs)} cell outputs, but only {len(actual_outputs)} found"
                    )

                if output_type == "stream":  # compare plain text
                    cls.compare_outputs_text(ref_out.get("text"), act_out.get("text"))

                else:
                    output_data = ref_out.get("data", {})

                    for mime, rule in {
                        "image/png": cls.compare_outputs_image,
                        "text/markdown": cls.compare_outputs_markup,
                        "text/html": cls.compare_outputs_html,
                    }.items():
                        if mime in output_data:
                            rule(output_data.get(mime), act_out.get("data", {}).get(mime))
                            break
                    else:  # unknown output type
                        raise NotImplementedError(
                            f"No rule to compare output type '{ref_out.get('output_type')}' with data '{ref_out.get('data', {}).keys()}'. \n"
                            "Currently only 'image/png', 'text/markdown' and 'text/html' are supported."
                        )

        except:
            for label, outputs in (
                ("Reference outputs:", reference_outputs),
                ("Actual outputs:", actual_outputs),
            ):
                print(label)
                for o in outputs:
                    print(o.get("output_type"))
                    print(o.get("text", ""))
                    print()

            raise

    @classmethod
    def compare_outputs_text(cls, reference_text, actual_text):
        """Plain text comparison"""
        test = TestCase()
        test.assertEqual(reference_text, actual_text)

    @classmethod
    def compare_outputs_markup(cls, ref_str, act_str):
        """Markup text comparison (ignoring whitespace changes)"""
        ref_str, act_str = [re.sub(r"\s+", " ", _) for _ in (ref_str, act_str)]
        TestCase().assertEqual(ref_str, act_str)

    @classmethod
    def compare_outputs_html(cls, ref_str, act_str):
        """HTML text comparison"""
        TestCase().assertEqual(ref_str, act_str)

    @classmethod
    def compare_outputs_image(cls, ref_img, act_img, *, tol=2):
        """Compare two base64 encoded images"""
        assert act_img, f"Expected image/png cell output, but {act_img} found"

        # save to temp file and compare
        with NamedTemporaryFile(suffix=".png") as rf, NamedTemporaryFile(suffix=".png") as af:
            rf.write(base64.b64decode(ref_img)) and rf.flush()
            af.write(base64.b64decode(act_img)) and af.flush()

            if diff := compare_images(rf.name, af.name, tol, True):
                os.makedirs("tests/output/", exist_ok=True)
                diff_image = shutil.copy(diff["diff"], "tests/output/")
                raise ValueError(
                    f"Image output differs from excpected output: RMS value {diff['rms']:g} > {diff['tol']:g} exceeds tolerance."
                    f"Diff image written to {diff_image}."
                )


def test_animations():
    t = NotebookTester(f"{dir}/animations.ipynb")
    t.execute(check_outputs=False)  # no good way of checking animation result


def test_colors():
    t = NotebookTester(f"{dir}/colors.ipynb")
    t.execute(check_outputs=True)


def test_hamiltonians():
    t = NotebookTester(f"{dir}/hamiltonians.ipynb")
    t.execute(check_outputs=True)


def test_line():
    t = NotebookTester(f"{dir}/line.ipynb")
    t.execute(check_outputs=True)


def test_phasespace():
    t = NotebookTester(f"{dir}/phasespace.ipynb")
    t.execute(check_outputs=True)


def test_concepts():
    t = NotebookTester(f"{dir}/concepts.ipynb")
    t.execute(check_outputs=True)


def test_timestructure():
    t = NotebookTester(f"{dir}/timestructure.ipynb")
    t.execute(check_outputs=True)


def test_twiss():
    t = NotebookTester(f"{dir}/twiss.ipynb")
    t.execute(check_outputs=True)


def test_utilities():
    t = NotebookTester(f"{dir}/utilities.ipynb")
    t.execute(check_outputs=True)
