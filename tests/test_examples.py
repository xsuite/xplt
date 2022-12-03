from testbook import testbook


@testbook(f"../examples/animations.ipynb")
def test_animations(tb):
    tb.execute()  # just confirm that the notebook runs


@testbook(f"../examples/colors.ipynb")
def test_colors(tb):
    tb.execute()  # just confirm that the notebook runs


@testbook(f"../examples/hamiltonians.ipynb")
def test_hamiltonians(tb):
    tb.execute()  # just confirm that the notebook runs


@testbook(f"../examples/phasespace.ipynb")
def test_phasespace(tb):
    tb.execute()  # just confirm that the notebook runs


@testbook(f"../examples/timestructure.ipynb")
def test_timestructure(tb):
    tb.execute()  # just confirm that the notebook runs


@testbook(f"../examples/twiss.ipynb")
def test_twiss(tb):
    tb.execute()  # just confirm that the notebook runs
