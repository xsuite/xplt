import os
from testbook import testbook

dir = "examples"
if not os.path.exists(dir):
    dir = os.path.join("..", dir)


@testbook(f"{dir}/animations.ipynb")
def test_animations(tb):
    tb.execute()  # just confirm that the notebook runs


@testbook(f"{dir}/colors.ipynb")
def test_colors(tb):
    tb.execute()  # just confirm that the notebook runs


@testbook(f"{dir}/hamiltonians.ipynb")
def test_hamiltonians(tb):
    tb.execute()  # just confirm that the notebook runs


@testbook(f"{dir}/line.ipynb")
def test_line(tb):
    tb.execute()  # just confirm that the notebook runs


@testbook(f"{dir}/phasespace.ipynb")
def test_phasespace(tb):
    tb.execute()  # just confirm that the notebook runs


@testbook(f"{dir}/concepts.ipynb")
def test_units(tb):
    tb.execute()  # just confirm that the notebook runs


@testbook(f"{dir}/timestructure.ipynb")
def test_timestructure(tb):
    tb.execute()  # just confirm that the notebook runs


@testbook(f"{dir}/twiss.ipynb")
def test_twiss(tb):
    tb.execute()  # just confirm that the notebook runs
