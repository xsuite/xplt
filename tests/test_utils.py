import pytest
import pytest_benchmark
import numpy as np
from numpy.testing import assert_equal
import xplt.util


@pytest.fixture(scope="function")
def benchmark_ref(request):
    # See https://github.com/ionelmc/pytest-benchmark/issues/166
    return pytest_benchmark.plugin.benchmark.__pytest_wrapped__.obj(request)


@pytest.mark.parametrize(
    "ndata,nbins,exp_time_fract",
    [
        (1_000, 2**17, 1 / 2),
        (1_000, 2**20, 1 / 4),
        (1_000_000, 2**20, 1 / 4),
        (1_000_000, 2**23, 1 / 10),
        (1_000_000, 2**27, 1 / 32),
    ],
)
def test_binned_data(benchmark, benchmark_ref, ndata, nbins, exp_time_fract):
    data = np.random.uniform(0, 1, ndata)

    v_min, dv, c_bin = benchmark(xplt.util.binned_data, data, n=nbins, v_range=(0, 1))
    c_ref = benchmark_ref(np.histogram, data, bins=nbins, range=(0, 1))[0]

    # test result
    assert v_min == 0
    assert dv == 1 / nbins
    assert np.sum(c_bin) == ndata
    assert c_bin.size == nbins
    assert_equal(c_bin, c_ref)

    # test performance
    fract = benchmark.stats.stats.median / benchmark_ref.stats.stats.median
    assert fract < exp_time_fract, (
        f"Performance of binned_data compared to np.histogram is {fract:g} but expected"
        f" {exp_time_fract} or less ({benchmark.stats.stats.median:g}s vs {benchmark_ref.stats.stats.median:g}s"
    )


def test_averaging():

    data = np.array([1, 2, 4, 5, 6, 8, 3, 10, 25, 13])

    avg = xplt.util.average(data, n=2)
    assert_equal(avg, [1.5, 4.5, 7, 6.5, 19])
    avg = xplt.util.average(data, n=2, keepdim=True)
    assert_equal(avg, [1.5, 1.5, 4.5, 4.5, 7, 7, 6.5, 6.5, 19, 19])
    avg = xplt.util.average(data, n=3, function=np.max)
    assert_equal(avg, [4, 8, 25])
