from numpy.testing import assert_equal
from packaging.version import Version
import xtrack as xt
import xplt.line


def test_nominal_and_effective_order():

    def assert_order(*test_params):
        """Check elements for nominal and effective order

        Args:
            *test_params: tuples with (element, nominal_order, effective_order)
        """
        for ele, no, eo in test_params:
            assert_equal(xplt.line.nominal_order(ele), no, f"Wrong nominal order for {ele}")
            assert_equal(xplt.line.effective_order(ele), eo, f"Wrong effective order for {ele}")

    # Elements with general knl and ksl coefficients
    for cls in (xt.Multipole,):
        assert_order(
            (cls(), 0, -1),
            (cls(length=1, knl=[]), 0, -1),
            (cls(length=1, knl=[1]), 0, 0),
            (cls(length=1, knl=[0, 1]), 1, 1),
            (cls(length=1, knl=[0, 0, 1]), 2, 2),
            (cls(length=1, knl=[0, 0, 0, 1]), 3, 3),
            (cls(length=1, ksl=[]), 0, -1),
            (cls(length=1, ksl=[1]), 0, 0),
            (cls(length=1, ksl=[0, 1]), 1, 1),
            (cls(length=1, ksl=[0, 0, 1]), 2, 2),
            (cls(length=1, ksl=[0, 0, 0, 1]), 3, 3),
        )

    if Version(xt.__version__) < Version("0.60"):
        return  # elements below require newer xtrack version

    # Thick elements
    assert_order(
        (xt.Bend(length=1, k0=1), 0, 0),
        (xt.Bend(length=1, k1=1), 0, 1),
        (xt.Quadrupole(length=1, k1=1), 1, 1),
        (xt.Quadrupole(length=1, k1s=1), 1, 1),
        (xt.Sextupole(length=1, k2=1), 2, 2),
        (xt.Sextupole(length=1, k2s=1), 2, 2),
        (xt.Octupole(length=1, k3=1), 3, 3),
        (xt.Octupole(length=1, k3s=1), 3, 3),
    )

    for cls, no in (
        (xt.Bend, 0),
        (xt.Quadrupole, 1),
        (xt.Sextupole, 2),
        (xt.Octupole, 3),
    ):
        assert_order(
            (cls(), no, -1),
            (cls(length=1, knl=[]), no, -1),
            (cls(length=1, knl=[1]), no, 0),
            (cls(length=1, knl=[0, 1]), no, 1),
            (cls(length=1, knl=[0, 0, 1]), no, 2),
            (cls(length=1, knl=[0, 0, 0, 1]), no, 3),
            (cls(length=1, ksl=[]), no, -1),
            (cls(length=1, ksl=[1]), no, 0),
            (cls(length=1, ksl=[0, 1]), no, 1),
            (cls(length=1, ksl=[0, 0, 1]), no, 2),
            (cls(length=1, ksl=[0, 0, 0, 1]), no, 3),
        )

    # Special thin elements
    assert_order(
        (xt.SimpleThinBend(knl=[0]), 0, -1),
        (xt.SimpleThinBend(knl=[1]), 0, 0),
        (xt.SimpleThinQuadrupole(knl=[0, 0]), 1, -1),
        (xt.SimpleThinQuadrupole(knl=[0, 1]), 1, 1),
    )


def test_repeated_elements_survey():

    # https://github.com/xsuite/xplt/issues/31
    line = xt.Line(elements={"obm": xt.Bend(length=0.5)}, element_names=["obm", "obm"])
    # line.replace_all_repeated_elements()
    survey = line.survey()
    print(survey)

    plot = xplt.FloorPlot(survey)

    boxes = plot.artists_boxes
    assert_equal(len(boxes), 2)
    assert_equal(boxes[0].get_center(), [0.25, 0])
    assert_equal(boxes[0].get_height(), 0.5)
    assert_equal(boxes[1].get_center(), [0.75, 0])
    assert_equal(boxes[1].get_height(), 0.5)


def test_sign_sticky():

    x = [0, 1, 5, 0, 5, -3, 0, 1, 0, -2, 0]
    assert_equal(xplt.line.sign_sticky(x, initial=1), [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1])
