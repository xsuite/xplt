#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting particle arrival times

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-24"

import types

import matplotlib as mpl
import numpy as np
import pint
import re

from .util import defaults
from .base import XManifoldPlot
from .particles import ParticlePlotMixin, ParticlesPlot
from .units import Prop


def binned_timeseries(times, n, what=None, range=None):
    """Get binned timeseries with equally spaced time bins

    From the particle arrival times (non-equally distributed timestamps), a timeseries with equally
    spaced time bins is derived. The time bin size is determined based on the number of bins.
    The parameter ``what`` determines what is returned for the timeseries. By default (what=None), the
    number of particles arriving within each time bin is returned. Alternatively, a particle property
    can be passed as array, in which case that property is averaged over all particles arriving within
    the respective bin (or 0 if no particles arrive within a time bin).

    Args:
        times (np.ndarray): Array of particle arrival times.
        n (int): Number of bins.
        what (np.ndarray | None): Array of associated data or None. Must have same shape as times. See above.
        range (tuple[int] | None): Tuple of (min, max) time values to consider. If None, the range is determined from the data.

    Returns:
        The timeseries as tuple (t_min, dt, values) where
        t_min is the start time of the timeseries data,
        dt is the time bin width and
        values are the values of the timeseries as array of length n.
    """

    # Note: The code below was optimized to run much faster than an ordinary
    # np.histogram, which quickly slows down for large datasets.
    # If you intend to change something here, make sure to benchmark it!

    t_min = np.min(times) if range is None or range[0] is None else range[0]
    t_max = np.max(times) if range is None or range[1] is None else range[1]
    dt = (t_max - t_min) / n
    # count timestamps in bins
    bins = ((times - t_min) / dt).astype(int)
    # bins are i*dt <= t < (i+1)*dt where i = 0 .. n-1
    mask = (bins >= 0) & (bins < n)  # igore times outside range
    bins = bins[mask]
    # count particles per time bin
    counts = np.bincount(bins, minlength=n)[:n]

    if what is None:
        # Return particle counts
        return t_min, dt, counts

    else:
        # Return 'what' averaged
        v = np.zeros(n)
        # sum up 'what' for all the particles in each bin
        np.add.at(v, bins, what[mask])
        # divide by particle count to get mean (default to 0)
        v[counts > 0] /= counts[counts > 0]
        return t_min, dt, v


class TimePlot(ParticlesPlot):
    def __init__(self, particles=None, kind="x+y", **kwargs):
        """
        A thin wrapper around the ParticlesPlot plotting data as function of time.
        For more information refer to the documentation of the :class:`~xplt.particles.ParticlesPlot` class.

        The plot is based on the particle arrival time, which is:
            - For circular lines: at_turn / frev - zeta / beta / c0
            - For linear lines: zeta / beta / c0

        Args:
            particles (Any): Particles data to plot.
            kind (str | list): Defines the properties to plot.
                This is a manifold subplot specification string like ``"x+y"``, see :class:`~.base.XManifoldPlot` for details.
                In addition, abbreviations for x-y-parameter pairs are supported (e.g. ``P`` for ``Px+Py``).
            kwargs: See :class:`~xplt.particles.ParticlesPlot` for more options.

        """
        super().__init__(particles, kind, as_function_of="t", **kwargs)


class TimeBinPlot(XManifoldPlot, ParticlePlotMixin):
    def __init__(
        self,
        particles=None,
        kind="count",
        *,
        bin_time=None,
        bin_count=None,
        exact_bin_time=True,
        relative=False,
        mask=None,
        plot_kwargs=None,
        **kwargs,
    ):
        """
        A binned histogram plot of particles as function of times.

        The plot is based on the particle arrival time, which is:
            - For circular lines: at_turn / frev - zeta / beta / c0
            - For linear lines: zeta / beta / c0

        The main purpose is to plot particle counts, but kind also accepts particle properties
        in which case the property is averaged over all particles falling into the bin.

        Useful to plot time structures of particles loss, such as spill structures.

        Args:
            particles (Any): Particles data to plot.
            kind (str | list): Defines the properties to plot, including 'count' (default), 'rate', 'cumulative', or a particle property to average.
                This is a manifold subplot specification string like ``"count-cumulative"``, see :class:`~.base.XManifoldPlot` for details.
                In addition, abbreviations for x-y-parameter pairs are supported (e.g. ``P`` for ``Px+Py``).
            bin_time (float): Time bin width if bin_count is None.
            bin_count (int): Number of bins if bin_time is None.
            exact_bin_time (bool): What to do if bin_time is given but length of data is not an exact multiple of it.
                If True, overhanging data is removed such that the data length is a multiple of bin_time.
                If False, bin_time is adjusted instead.
            relative (bool): If True, plot relative numbers normalized to total count.
                If what is a particle property, this has no effect.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            plot_kwargs (dict): Keyword arguments passed to the plot function.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments

        """
        kwargs = self._init_particle_mixin(**kwargs)
        kwargs["data_units"] = defaults(
            kwargs.get("data_units"),
            count=Prop("$N$", unit="1", description="Particles per bin"),
            cumulative=Prop("$N$", unit="1", description="Particles (cumulative)"),
            rate=Prop("$\\dot{N}$", unit="1/s", description="Particle rate"),
        )
        super().__init__(
            on_x="t",
            on_y=kind,
            **kwargs,
        )

        if bin_time is None and bin_count is None:
            bin_count = 100
        self.bin_time = bin_time
        self.bin_count = bin_count
        self.exact_bin_time = exact_bin_time
        self.relative = relative

        # Format plot axes
        self.axis(-1).set(xlabel=self.label_for("t"), ylim=(0, None))
        if self.relative:
            for a in self.axflat:
                a.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

        # Create plot elements
        def create_artists(i, j, k, ax, p):
            kwargs = defaults(plot_kwargs, lw=1, label=self._legend_label_for(p))
            if p in ("count", "rate", "cumulative"):
                kwargs = defaults(kwargs, drawstyle="steps-pre")
            return ax.plot([], [], **kwargs)[0]

        self._create_artists(create_artists)

        # set data
        if particles is not None:
            self.update(particles, mask=mask, autoscale=True)

    def _get_property(self, p):
        prop = super()._get_property(p)
        if prop.key not in ("count", "rate", "cumulative", "t"):
            # it is averaged
            prop.symbol = f"$\\langle${prop.symbol}$\\rangle$"
        return prop

    def update(self, particles, mask=None, autoscale=False):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (bool): Whether or not to perform autoscaling on all axes.

        Returns:
            list: Changed artists
        """

        # extract times
        times = self._get_masked(particles, "t", mask)

        # re-sample times into equally binned time series
        if self.bin_count:
            n = self.bin_count
            t_range = None
        elif self.exact_bin_time:
            n = int((np.max(times) - np.min(times)) / self.bin_time)
            t_range = np.min(times) + np.array([0, n * self.bin_time])
        else:
            n = int(round((np.max(times) - np.min(times)) / self.bin_time))
            t_range = None

        # update plots
        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                count_based = False
                for k, p in enumerate(pp):
                    prop = self._get_property(p)
                    count_based = prop.key in ("count", "rate", "cumulative")
                    if count_based:
                        property = None
                    else:
                        property = self._get_masked(particles, prop.key, mask)

                    t_min, dt, timeseries = binned_timeseries(times, n, property, t_range)
                    timeseries = timeseries.astype(np.float64)
                    edges = np.linspace(t_min, t_min + dt * n, n + 1)

                    self.annotate(
                        f'$\\Delta t_\\mathrm{{bin}} = {pint.Quantity(dt, "s").to_compact():~.4L}$'
                    )

                    if self.relative:
                        if not count_based:
                            raise ValueError(
                                "Relative plots are only supported for kind 'count', 'rate' or 'cumulative'."
                            )
                        timeseries /= len(times)

                    if prop.key == "rate":
                        timeseries /= dt

                    # target units
                    edges *= self.factor_for("t")
                    timeseries *= self.factor_for(p)

                    # update plot
                    if prop.key == "cumulative":
                        # steps open after last bin
                        timeseries = np.concatenate(([0], np.cumsum(timeseries)))
                    elif count_based:
                        # steps go back to zero after last bin
                        edges = np.append(edges, edges[-1])
                        timeseries = np.concatenate(([0], timeseries, [0]))
                    else:
                        edges = (edges[1:] + edges[:-1]) / 2
                    self.artists[i][j][k].set_data((edges, timeseries))
                    changed.append(self.artists[i][j][k])

                if autoscale:
                    a = self.axis(i, j)
                    a.relim()
                    a.autoscale()
                    if count_based:
                        a.set(ylim=(0, None))

        return changed


class TimeFFTPlot(XManifoldPlot, ParticlePlotMixin):
    def __init__(
        self,
        particles=None,
        kind="count",
        *,
        fmax=None,
        relative=False,
        log=None,
        scaling=None,
        mask=None,
        plot_kwargs=None,
        **kwargs,
    ):
        """
        A frequency plot based on particle arrival times.

        The particle arrival time is:
            - For circular lines: at_turn / frev - zeta / beta / c0
            - For linear lines: zeta / beta / c0

        From the particle arrival times (non-equally distributed timestamps), a timeseries with equally
        spaced time bins is derived. The time bin size is determined based on fmax and performance considerations.
        By default, the binned property is the number of particles arriving within the bin time (what='count').
        Alternatively, a particle property may be specified (e.g. what='x'), in which case that property is
        averaged over all particles arriving in the respective bin. The FFT is then computed over the timeseries.

        Useful to plot time structures of particles loss, such as spill structures.

        Args:
            particles (Any): Particles data to plot.
            kind (str | list): Defines the properties to make the FFT over, including 'count' (default), or a particle property to average.
                This is a manifold subplot specification string like ``"count-cumulative"``, see :class:`~.base.XManifoldPlot` for details.
                In addition, abbreviations for x-y-parameter pairs are supported (e.g. ``P`` for ``Px+Py``).
            fmax (float): Maximum frequency (in Hz) to plot.
            relative (bool): If True, plot relative frequencies (f/frev) instead of absolute frequencies (f).
            log (bool): If True, plot on a log scale.
            scaling (str | dict): Scaling of the FFT. Can be 'amplitude' or 'pds' or a dict with a scaling per property.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            plot_kwargs (dict): Keyword arguments passed to the plot function.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments

        """

        self._fmax = fmax
        self.relative = relative
        self._scaling = scaling
        if log is None:
            log = not relative

        kwargs = self._init_particle_mixin(
            **kwargs,
        )
        kwargs["data_units"] = defaults(
            kwargs.get("data_units"),
            count=Prop("N", unit="1", description="Particles per bin"),
        )
        super().__init__(
            on_x="t",
            on_y=kind,
            **kwargs,
        )

        # Format plot axes
        self.axis(-1).set(
            xlabel="$f/f_{rev}$" if self.relative else self.label_for("f"),
        )
        for a in self.axflat:
            a.set(ylim=(0, None))
            if log:
                a.set(
                    xscale="log",
                    yscale="log",
                )

        # Create plot elements
        def create_artists(i, j, k, ax, p):
            kwargs = defaults(plot_kwargs, lw=1, label=self._legend_label_for(p))
            return ax.plot([], [], **kwargs)[0]

        self._create_artists(create_artists)

        # set data
        if particles is not None:
            self.update(particles, mask=mask, autoscale=True)

    def _get_scaling(self, key):
        if isinstance(self._scaling, str):
            return self._scaling.lower()
        if isinstance(self._scaling, dict) and key in self._scaling:
            return self._scaling[key].lower()
        return "pds" if key == "count" else "amplitude"

    def fmax(self, particles):
        if self._fmax is not None:
            return self._fmax
        if self.relative:
            return self.frev(particles)
        raise ValueError("fmax must be specified.")

    def update(self, particles, mask=None, autoscale=False):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (bool): Whether or not to perform autoscaling on all axes.

        Returns:
            list: Changed artists
        """

        # extract times and associated property
        times = self._get_masked(particles, "t", mask)

        # re-sample times into equally binned time series
        fmax = self.fmax(particles)
        n = int(np.ceil((np.max(times) - np.min(times)) * fmax * 2))
        # to improve FFT performance, round up to next power of 2
        self.nbins = n = 1 << (n - 1).bit_length()

        # update plots
        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):
                    prop = self._get_property(p)
                    count_based = prop.key == "count"

                    if count_based:
                        property = None
                    else:
                        property = self._get_masked(particles, prop.key, mask)

                    # compute binned timeseries
                    t_min, dt, timeseries = binned_timeseries(times, n, property)

                    # calculate fft without DC component
                    freq = np.fft.rfftfreq(n, d=dt)[1:]
                    if self.relative:
                        freq /= self.frev(particles)
                    else:
                        freq *= self.factor_for("f")
                    mag = np.abs(np.fft.rfft(timeseries))[1:]
                    if self._get_scaling(prop.key) == "amplitude":
                        # amplitude in units of p
                        mag *= 2 / len(timeseries) * self.factor_for(p)
                    elif self._get_scaling(prop.key) == "pds":
                        # power density spectrum in a.u.
                        mag = mag**2

                    # update plot
                    self.artists[i][j][k].set_data(freq, mag)
                    changed.append(self.artists[i][j][k])

                if autoscale:
                    a = self.axis(i, j)
                    a.relim()
                    a.autoscale()
                    log = a.get_xscale() == "log"
                    xlim = np.array((10.0, fmax) if log else (0.0, fmax))
                    if self.relative:
                        xlim /= self.frev(particles)
                    else:
                        xlim *= self.factor_for("f")
                    a.set_xlim(xlim)
                    if a.get_yscale() != "log":
                        a.set_ylim(0, None)

        self.annotate(f"$\\Delta t_\\mathrm{{bin}} = {pint.Quantity(dt, 's').to_compact():~.4L}$")

        return changed

    def _get_property(self, p):
        prop = super()._get_property(p)
        if p not in "f":
            # it is the FFT of it
            sym = prop.symbol.strip("$")
            if self._get_scaling(prop.key) == "amplitude":
                prop.symbol = f"$\\hat{{{sym}}}$"
            elif self._get_scaling(prop.key) == "pds":
                prop.symbol = f"$|\\mathrm{{FFT({sym})}}|^2$"
                prop.unit = "a.u."  # arbitrary unit
            else:
                prop.symbol = f"$|\\mathrm{{FFT({sym})}}|$"
                prop.unit = "a.u."  # arbitrary unit
        return prop

    def plot_harmonics(self, f, df=0, *, n=20, **plot_kwargs):
        """Add vertical lines or spans indicating the location of values or spans and their harmonics

        Args:
            f (float or list of float): Fundamental frequency or list of frequencies in Hz.
            df (float or list of float, optional): Bandwidth or list of bandwidths centered around frequencies(s) in Hz.
            n (int): Number of harmonics to plot.
            plot_kwargs: Keyword arguments to be passed to plotting method
        """
        for a in self.axflat:
            super().plot_harmonics(
                a, self.factor_for("f") * f, self.factor_for("f") * df, n=n, **plot_kwargs
            )


class TimeIntervalPlot(XManifoldPlot, ParticlePlotMixin):
    def __init__(
        self,
        particles=None,
        *,
        tmax=None,
        bin_time=None,
        bin_count=None,
        exact_bin_time=True,
        log=True,
        mask=None,
        plot_kwargs=None,
        **kwargs,
    ):
        """
        A histogram plot of particle arrival intervals (i.e. delay between consecutive particles).

        The plot is based on the particle arrival time, which is:
            - For circular lines: at_turn / frev - zeta / beta / c0
            - For linear lines: zeta / beta / c0

        Useful to plot time structures of particles loss, such as spill structures.

        Args:
            particles (Any): Particles data to plot.
            tmax (float): Maximum interval (in s) to plot.
            bin_time (float): Time bin width if bin_count is None.
            bin_count (int): Number of bins if bin_time is None.
            exact_bin_time (bool): What to do if bin_time is given but tmax is not an exact multiple of it.
                If True, tmax is adjusted to be a multiple of bin_time.
                If False, bin_time is adjusted instead.
            log (bool): If True, plot on a log scale.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            plot_kwargs (dict): Keyword arguments passed to the plot function.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments


        """
        if tmax is None:
            raise ValueError("tmax must be specified.")

        if bin_time is not None:
            if exact_bin_time:
                tmax = bin_time * round(tmax / bin_time)
            else:
                bin_time = tmax / round(tmax / bin_time)

        kwargs = self._init_particle_mixin(**kwargs)
        kwargs["data_units"] = defaults(
            kwargs.get("data_units"),
            dt=Prop("$\\Delta t$", unit="s", description="Delay between consecutive particles"),
            count=Prop("$N$", unit="1", description="Particles per bin"),
        )
        super().__init__(
            on_x="dt",
            on_y="count",
            **kwargs,
        )

        if bin_time is None and bin_count is None:
            bin_count = 100
        self._bin_time = bin_time
        self._bin_count = bin_count
        self.tmax = tmax

        # create plot elements
        def create_artists(i, j, k, ax, p):
            return ax.step([], [], **defaults(plot_kwargs, lw=1))[0]

        self._create_artists(create_artists)

        # Format plot axes
        ax = self.axis(-1)
        ax.set(xlim=(self.bin_time if log else 0, self.tmax * self.factor_for("t")))
        if log:
            ax.set(xscale="log", yscale="log")
        else:
            ax.set(ylim=(0, None))

        # set data
        if particles is not None:
            self.update(particles, mask=mask, autoscale=True)

    @property
    def bin_time(self):
        return self._bin_time or self.tmax / self._bin_count

    @property
    def bin_count(self):
        return int(np.ceil(self.tmax / self.bin_time))

    def update(self, particles, mask=None, autoscale=False):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (bool): Whether or not to perform autoscaling on all axes.
        """

        # extract times
        times = self._get_masked(particles, "t", mask)
        delay = self.factor_for("t") * np.diff(sorted(times))

        # calculate and plot histogram
        counts, edges = np.histogram(
            delay, bins=self.bin_count, range=(0, self.bin_count * self.bin_time)
        )
        steps = (np.append(edges, edges[-1]), np.concatenate(([0], counts, [0])))

        self.annotate(
            f"$\\Delta t_\\mathrm{{bin}} = {pint.Quantity(self.bin_time, 's').to_compact():~.4L}$"
        )

        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                ax = self.axis(i, j)
                for k, p in enumerate(pp):
                    if p != "count":
                        raise ValueError(f"Invalid plot parameter {p}.")
                    self.artists[i][j][k].set_data(steps)

                if autoscale:
                    ax.relim()
                    ax.autoscale()
                    if not ax.get_yscale() == "log":
                        ax.set(ylim=(0, None))

    def plot_harmonics(self, t, *, n=20, **plot_kwargs):
        """Add vertical lines or spans indicating the location of values or spans and their harmonics

        Args:
            t (float or list of float): Period in s.
            n (int): Number of harmonics to plot.
            plot_kwargs: Keyword arguments to be passed to plotting method
        """
        for a in self.axflat:
            super().plot_harmonics(a, self.factor_for("t") * t, n=n, **plot_kwargs)


class MetricesMixin:
    """Mixin to evaluate particle fluctuation metrices for spill quality analysis

    The following metrics are implemented:
        cv: Coefficient of variation
            cv = std(N)/mean(N)
        duty: Spill duty factor
            F = mean(N)**2 / mean(N**2)
        maxmean: Maximum to mean ratio
            M = max(N) / mean(N)

    """

    _metric_properties = dict(
        cv=Prop("$c_v=\\sigma/\\mu$", unit="1", description="Coefficient of variation"),
        duty=Prop(
            "$F=\\langle N \\rangle^2/\\langle N^2 \\rangle$",
            unit="1",
            description="Spill duty factor",
        ),
        maxmean=Prop(
            "$M=\\hat{N}/\\langle N \\rangle$", unit="1", description="Max-to-mean ratio"
        ),
    )

    @staticmethod
    def _calculate_metric(N, metric, axis=None):
        """Calculate the metric over the array N"""
        if metric == "cv":
            Cv = np.std(N, axis=axis) / np.mean(N, axis=axis)
            Cv_poisson = 1 / np.mean(N, axis=axis) ** 0.5
            return Cv, Cv_poisson
        elif metric == "duty":
            F = np.mean(N, axis=axis) ** 2 / np.mean(N**2, axis=axis)
            F_poisson = 1 / (1 + 1 / np.mean(N, axis=axis))
            return F, F_poisson
        elif metric == "maxmean":
            M = np.max(N, axis=axis) / np.mean(N, axis=axis)
            return M, np.nan * np.empty_like(M)
        else:
            raise ValueError(f"Unknown metric {metric}")

    def _format_metric_axes(self, add_compatible_twin_axes):
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                a = self.axis(i, j)
                if np.all(np.array(pp) == "duty"):
                    a.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

            # indicate compatible metric on opposite axis (if free space)
            if add_compatible_twin_axes and len(ppp) == 1:
                if np.all(np.array(ppp[0]) == "duty"):
                    other = "cv"
                    formatter = lambda du, i: "∞" if du <= 0 else f"{abs(1/du-1)**0.5:.1f}"
                elif np.all(np.array(ppp[0]) == "cv"):
                    other = "duty"
                    formatter = lambda cv, i: "" if cv < 0 else f"{100/(1+cv**2):.1f} %"
                else:
                    other = None

                if other is not None:
                    a = self.axis(i)
                    at = a.twinx()
                    at.set(ylabel=self.label_for(other), ylim=a.get_ylim())
                    at.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter))
                    a.callbacks.connect("ylim_changed", lambda a: at.set(ylim=a.get_ylim()))


class TimeVariationPlot(XManifoldPlot, ParticlePlotMixin, MetricesMixin):
    def __init__(
        self,
        particles=None,
        metric="cv",
        *,
        counting_dt=None,
        counting_bins=None,
        evaluate_dt=None,
        evaluate_bins=None,
        poisson=True,
        mask=None,
        plot_kwargs=None,
        **kwargs,
    ):
        """
        Plot variability of particle time on microscopic scale as function of time on macroscopic scale

        The particle arrival times are histogramed into counting bins, the width of which
        corresponds to the time resolution of a detector (``counting_dt``).
        The plot estimates fluctuations in these particle counts by applying a metric
        over an evaluation window (``evaluation_dt``).

        See :class:`~.timestructure.MetricesMixin` for a list of implemented metrics.

        If the particle data corresponds to particles lost at the extraction septum,
        the plot yields the spill quality as function of extraction time.

        Args:
            particles (Any): Particles data to plot.
            metric (str | list): Metric to plot. See above for list of implemented metrics.
            counting_dt (float): Time bin width for counting if counting_bins is None.
            counting_bins (int): Number of bins if counting_dt is None.
            evaluate_dt (float): Time bin width for metric evaluation if evaluate_bins is None.
            evaluate_bins (int): Number of bins if evaluate_dt is None.
            poisson (bool): If true, indicate poisson limit.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            plot_kwargs (dict): Keyword arguments passed to the plot function.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments

        """
        kwargs = self._init_particle_mixin(
            **kwargs,
        )
        kwargs["data_units"] = defaults(
            kwargs.get("data_units"),
            **self._metric_properties,
        )
        super().__init__(
            on_x="t",
            on_y=metric,
            **kwargs,
        )

        if counting_dt is None and counting_bins is None:
            counting_bins = 100 * 100
        if evaluate_dt is None and evaluate_bins is None:
            evaluate_bins = 100
        self.counting_dt = counting_dt
        self.counting_bins = counting_bins
        self.evaluate_dt = evaluate_dt
        self.evaluate_bins = evaluate_bins

        # Format plot axes
        self._format_metric_axes(kwargs.get("ax") is None)

        # Create plot elements
        def create_artists(i, j, k, ax, p):
            kwargs = defaults(plot_kwargs, lw=1, label=self._legend_label_for(p))
            step = ax.step([], [], **kwargs)[0]
            if poisson:
                kwargs.update(
                    color=step.get_color() or "gray",
                    alpha=0.5,
                    zorder=1.9,
                    lw=1,
                    ls=":",
                    label="Poisson limit",
                )
                pstep = ax.step([], [], **kwargs)[0]
            else:
                pstep = None
            return step, pstep

        self._create_artists(create_artists)

        # set data
        if particles is not None:
            self.update(particles, mask=mask, autoscale=True)

    def update(self, particles, mask=None, autoscale=False):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (bool): Whether or not to perform autoscaling on all axes.

        Returns:
            Changed artists
        """

        # extract times
        times = self._get_masked(particles, "t", mask)

        # re-sample times into equally binned time series
        ncbins = self.counting_bins or int(
            np.ceil((np.max(times) - np.min(times)) / self.counting_dt)
        )
        if self.evaluate_bins is not None:
            nebins = int(ncbins / self.evaluate_bins)
        else:
            nebins = int(ncbins * self.evaluate_dt / (np.max(times) - np.min(times)))

        # bin into counting bins
        t_min, dt, counts = binned_timeseries(times, ncbins)
        edges = np.linspace(t_min, t_min + dt * ncbins, ncbins + 1)

        # make 2D array by subdividing into evaluation bins
        N = counts[: int(len(counts) / nebins) * nebins].reshape((-1, nebins))
        E = edges[: int(len(edges) / nebins + 1) * nebins : nebins]

        self.annotate(
            f'$\\Delta t_\\mathrm{{count}} = {pint.Quantity(dt, "s"):#~.4L}$\n'
            f'$\\Delta t_\\mathrm{{evaluate}} = {pint.Quantity(dt*nebins, "s"):#~.4L}$'
        )

        # update plots
        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):

                    # calculate metrics
                    F, F_poisson = self._calculate_metric(N, p, axis=1)

                    # update plot
                    step, pstep = self.artists[i][j][k]
                    edges = self.factor_for("t") * np.append(E, E[-1])
                    steps = np.concatenate(([0], F, [0]))
                    step.set_data((edges, steps))
                    changed.append(step)
                    if pstep:
                        steps = np.concatenate(([0], F_poisson, [0]))
                        pstep.set_data((edges, steps))
                        changed.append(pstep)

                if autoscale:
                    a = self.axis(i, j)
                    a.relim()
                    a.autoscale()
        return changed


class TimeVariationScalePlot(XManifoldPlot, ParticlePlotMixin, MetricesMixin):
    def __init__(
        self,
        particles=None,
        kind="cv",
        *,
        time_range=None,
        counting_dt_min=None,
        counting_dt_max=None,
        counting_bins_per_evaluation=50,
        poisson=True,
        mask=None,
        log=True,
        plot_kwargs=None,
        **kwargs,
    ):
        """Plot variability of particle time as function of timescale

        The particle arrival times are histogramed into counting bins, the width of which
        corresponds to the time resolution of a detector (``counting_dt``).
        The plot estimates fluctuations in these particle counts by applying a metric
        over an evaluation window (``counting_bins_per_evaluation*counting_dt``).

        See :class:`~.timestructure.MetricesMixin` for a list of implemented metrics.

        If the particle data corresponds to particles lost at the extraction septum,
        the plot yields the spill quality on different timescales.


        Args:
            particles (Any): Particles data to plot.
            kind (str | list): Metric to plot. See above for list of implemented metrics.
            time_range (tuple): Time range of particles to consider. If None, all particles are considered.
            counting_dt_min (float): Minimum time bin width for counting.
            counting_dt_max (float): Maximum time bin width for counting.
            counting_bins_per_evaluation (int): Number of counting bins used to evaluate metric over.
                Use None to evaluate metric once over all bins. Otherwise, the metric is evaluated
                over each ``counting_bins_per_evaluation`` consecutive bins, and average and std of
                all evaluations plotted. This suppresses fluctuations on timescales > 100*t_bin
                to influence the metric.
            poisson (bool): Whether or not to plot the Poisson limit.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            log (bool): Whether or not to plot the x-axis in log scale.
            plot_kwargs (dict): Keyword arguments passed to the plot function.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments


        """
        kwargs = self._init_particle_mixin(
            **kwargs,
        )
        kwargs["data_units"] = defaults(
            kwargs.get("data_units"),
            tbin=Prop("$\\Delta t_\\mathrm{count}$", unit="s", description="Time resolution"),
            **self._metric_properties,
        )
        super().__init__(
            on_x="tbin",
            on_y=kind,
            **kwargs,
        )

        self.time_range = time_range
        self.counting_dt_min = counting_dt_min
        self.counting_dt_max = counting_dt_max
        self.counting_bins_per_evaluation = counting_bins_per_evaluation
        self.log = log

        # Format plot axes
        self._format_metric_axes(kwargs.get("ax") is None)
        for a in self.axflat:
            a.set(xscale="log" if self.log else "lin")

        # Create plot elements
        def create_artists(i, j, k, ax, p):
            kwargs = defaults(plot_kwargs, label=self._legend_label_for(p))
            plot = ax.plot([], [], **kwargs)[0]
            kwargs.update(color=plot.get_color())
            if self.counting_bins_per_evaluation:
                self._errkw = kwargs.copy()
                self._errkw.update(zorder=1.8, alpha=0.3, ls="-", lw=0)
                errorbar = ax.fill_between([], [], [], **self._errkw)
            else:
                errorbar = None
            if poisson:
                kwargs.update(zorder=1.9, ls=":", label="Poisson limit")
                pstep = ax.plot([], [], **kwargs)[0]
            else:
                pstep = None
            return [plot, errorbar, pstep]

        self._create_artists(create_artists)

        # legend with combined patch
        if self.counting_bins_per_evaluation:
            # merge plot and errorbar patches
            for i, h in enumerate(self._legend_entries):
                labels = [h[0].get_label()] + [_.get_label() for _ in h[2:]]
                self._legend_entries[i] = [tuple(h[0:2])] + h[2:]
                self.legend(i, show="auto", labels=labels)

        # set data
        if particles is not None:
            self.update(particles, mask=mask, autoscale=True)

    def update(self, particles, mask=None, autoscale=False):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (bool): Whether or not to perform autoscaling on all axes.

        Returns:
            Changed artists
        """

        # extract times
        times = self._get_masked(particles, "t", mask)
        if self.time_range:
            times = times[(self.time_range[0] < times) & (times < self.time_range[1])]
        ntotal = times.size
        duration = np.max(times) - np.min(times)

        # annotate plot
        # f'$\\langle\\dot{{N}}\\rangle = {pint.Quantity(ntotal/duration, "1/s"):~.4L}$\n'
        self.annotate(
            "$\\Delta t_\\mathrm{evaluate} = "
            + (
                f"{self.counting_bins_per_evaluation:g}\\,\\Delta t_\\mathrm{{count}}$"
                if self.counting_bins_per_evaluation
                else f"{pint.Quantity(duration, 's'):#~.4L}$"
            )
        )

        # determine timescales
        if self.counting_dt_min is None:
            ncbins_max = int(ntotal / 50)  # at least 50 particles per bin (on average)
        else:
            ncbins_max = int(duration / self.counting_dt_min + 1)

        if self.counting_dt_max is None:
            ncbins_min = 50  # at least 50 bins to calculate metric
        else:
            ncbins_min = int(duration / self.counting_dt_max + 1)

        if ncbins_min > ncbins_max or ntotal < 1e4:
            print(
                f"Warning: Data length ({duration:g} s), counting_dt_min ({duration/ncbins_max:g} s), "
                f"counting_dt_max ({duration/ncbins_min:g} s) and/or count ({ntotal:g}) insufficient. "
                f"Nothing plotted."
            )
            return

        if self.log:
            ncbins_arr = np.unique(
                (1 / np.geomspace(1 / ncbins_min, 1 / ncbins_max, 100)).astype(int)
            )
        else:
            ncbins_arr = np.unique(
                (1 / np.linspace(1 / ncbins_min, 1 / ncbins_max, 100)).astype(int)
            )

        # compute metrices
        DT = np.empty(ncbins_arr.size)
        F = {m: np.empty_like(DT) for m in self.on_y_unique}
        F_std = {m: np.empty_like(DT) for m in self.on_y_unique}
        F_poisson = {m: np.empty_like(DT) for m in self.on_y_unique}
        for i, nbin in enumerate(ncbins_arr):
            _, DT[i], N = binned_timeseries(times, nbin)

            # calculate metric on sliding window
            stride = min(self.counting_bins_per_evaluation or N.size, N.size)
            N = np.lib.stride_tricks.sliding_window_view(N, stride)

            for metric in F.keys():
                # calculate metrics
                v, lim = self._calculate_metric(N, metric, axis=1)
                F[metric][i] = np.mean(v)
                F_std[metric][i] = np.std(v) or np.nan
                F_poisson[metric][i] = np.mean(lim)

        DT = DT * self.factor_for(self.on_x)

        # update plots
        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                ax = self.axis(i, j)
                for k, p in enumerate(pp):
                    plot, errorbar, pstep = self.artists[i][j][k]

                    # update plot
                    plot.set_data((DT, F[p]))
                    changed.append(plot)
                    if errorbar:
                        changed.append(errorbar)
                        errorbar.remove()
                        errorbar = ax.fill_between(
                            DT, F[p] - F_std[p], F[p] + F_std[p], **self._errkw
                        )
                        self.artists[i][j][k][1] = errorbar
                        changed.append(errorbar)
                    if pstep:
                        pstep.set_data((DT, F_poisson[p]))
                        changed.append(pstep)

                if autoscale:
                    self._autoscale(ax, self.artists[i][j])

        return changed


## Restrict star imports to local namespace
__all__ = [
    name
    for name, thing in globals().items()
    if not (name.startswith("_") or isinstance(thing, types.ModuleType))
]
