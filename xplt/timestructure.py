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
from .base import XManifoldPlot, TwinFunctionLocator, TransformedLocator
from .particles import ParticlePlotMixin, ParticlesPlot
from .units import Prop


def binned_timeseries(times, *, what=None, n=None, dt=None, t_range=None, moments=1):
    """Get binned timeseries with equally spaced time bins

    From the particle arrival times (non-equally distributed timestamps), a timeseries with equally
    spaced time bins is derived. The parameter ``what`` determines what is returned for the timeseries.
    By default (what=None), the number of particles arriving within each time bin is returned.
    Alternatively, a particle property can be passed as array, in which case that property is averaged
    over all particles arriving within the respective bin (or 0 if no particles arrive within a time bin).
    It is also possible to specify the moments to return, i.e. the power to which the property is raised
    before averaging. This allows to determine mean (1st moment, default) and variance (difference between
    2nd and 1st moment) etc. To disable averaging, pass None as the moment

    Args:
        times (np.ndarray): Array of particle arrival times.
        n (int | None): Number of bins. Must not be used together with dt.
        dt (float | None): Bin width in seconds. Must not be used together with n.
        t_range (tuple[float] | None): Tuple of (min, max) time values to consider. If None, the range is determined from the data.
        what (np.ndarray | None): Array of associated data or None. Must have same shape as times. See above.
        moments (int | list[int | None] | None): The moment(s) to return for associated data if what is not None. See above.

    Returns:
        The timeseries as tuple (t_min, dt, values) where
        t_min is the start time of the timeseries data,
        dt is the time bin width and
        values are the values of the timeseries as array of length n.
    """

    t_min = np.min(times) if t_range is None or t_range[0] is None else t_range[0]
    t_max = np.max(times) if t_range is None or t_range[1] is None else t_range[1]

    if n is not None and dt is None:
        # number of bins requested, adjust bin width accordingly
        dt = (t_max - t_min) / n
    elif n is None and dt is not None:
        # bin width requested, adjust number of bins accordingly
        n = int(np.ceil((t_max - t_min) / dt))
    else:
        raise ValueError(f"Exactly one of n or dt must be specified, but got n={n} and dt={dt}")

    # Note: The code below was optimized to run much faster than an ordinary
    # np.histogram, which quickly slows down for large datasets.
    # If you intend to change something here, make sure to benchmark it!

    # count timestamps in bins
    bins = np.floor((times - t_min) / dt).astype(int)
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
        result = [t_min, dt]
        if isinstance(moments, int) or moments is None:
            moments = [moments]
        for m in moments:
            v = np.zeros(n)
            # sum up 'what' for all the particles in each bin
            power = m if m is not None else 1
            np.add.at(v, bins, what[mask] ** power)
            if m is not None:
                # divide by particle count to get mean (default to 0)
                v[counts > 0] /= counts[counts > 0]
            result.append(v)
        return result


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
        relative=False,
        moment=1,
        mask=None,
        time_range=None,
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
            bin_time (float): Time bin width (in s) if bin_count is None.
            bin_count (int): Number of bins if bin_time is None.
            relative (bool): If True, plot relative numbers normalized to total count.
                If what is a particle property, this has no effect.
            moment (int): The moment(s) to plot if kind is a particle property.
                Allows to get the mean (1st moment, default), variance (difference between 2nd and 1st moment) etc.

            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            time_range (tuple): Time range of particles to consider. If None, all particles are considered.
            plot_kwargs (dict): Keyword arguments passed to the plot function.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments

        """
        self.moment = moment
        kwargs = self._init_particle_mixin(**kwargs)
        kwargs["data_units"] = defaults(
            kwargs.get("data_units"),
            count=Prop("$N$", unit="1", description="Particles per bin"),
            cumulative=Prop("$N$", unit="1", description="Particles (cumulative)"),
            rate=Prop("$\\dot{N}$", unit="1/s", description="Particle rate"),
            charge=Prop("$Q$", unit=Prop.get("q0").unit, description="Charge per bin"),
            current=Prop("$I$", unit=f"({Prop.get('q0').unit})/s", description="Current"),
        )
        kwargs["display_units"] = defaults(
            kwargs.get("display_units"),
            current="nA",
        )
        super().__init__(
            on_x="t",
            on_y=kind,
            **kwargs,
        )

        if bin_time is None and bin_count is None:
            bin_count = 100
        if bin_time is not None and bin_count is not None:
            raise ValueError("Only one of bin_time or bin_count may be specified.")
        self.bin_time = bin_time
        self.bin_count = bin_count
        self.relative = relative
        self.time_range = time_range

        # Format plot axes
        self.axis(-1).set(xlabel=self.label_for("t"), ylim=(0, None))
        if self.relative:
            for a in self.axflat:
                a.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

        # Create plot elements
        def create_artists(i, j, k, ax, p):
            kwargs = defaults(plot_kwargs, lw=1, label=self._legend_label_for(p))
            if self._count_based(p):
                kwargs = defaults(kwargs, drawstyle="steps-pre")
            return ax.plot([], [], **kwargs)[0]

        self._create_artists(create_artists)

        # set data
        if particles is not None:
            self.update(particles, mask=mask, autoscale=True)

    def _count_based(self, key):
        return key in ("count", "rate", "cumulative", "charge", "current")

    def _get_property(self, p):
        prop = super()._get_property(p)
        if prop.key != "t" and self.moment is not None and not self._count_based(prop.key):
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

        # update plots
        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                count_based = False
                for k, p in enumerate(pp):
                    prop = self._get_property(p)
                    count_based = self._count_based(prop.key)
                    if prop.key in ("current", "charge"):
                        property = self._get_masked(particles, "q", mask)
                    elif count_based:
                        property = None
                    else:
                        property = self._get_masked(particles, prop.key, mask)

                    t_min, dt, timeseries = binned_timeseries(
                        times,
                        what=property,
                        n=self.bin_count,
                        dt=self.bin_time,
                        t_range=self.time_range,
                        moments=None if count_based else self.moment,
                    )
                    timeseries = timeseries.astype(np.float64)
                    edges = np.linspace(t_min, t_min + dt * timeseries.size, timeseries.size + 1)

                    self.annotate(
                        f'$\\Delta t_\\mathrm{{bin}} = {pint.Quantity(dt, "s").to_compact():~.4L}$'
                    )

                    if self.relative:
                        if not count_based:
                            raise ValueError(
                                "Relative plots are only supported for kind 'count', 'rate', 'cumulative', 'charge' or 'current'."
                            )
                        timeseries /= len(times)

                    if prop.key in ("rate", "current"):
                        timeseries /= dt

                    # target units
                    edges *= self.factor_for("t")
                    timeseries *= self.factor_for(p)

                    # expression wrappers
                    timeseries = prop.evaluate_expression(timeseries)
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
        time_range=None,
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
            time_range (tuple): Time range of particles to consider. If None, all particles are considered.
            plot_kwargs (dict): Keyword arguments passed to the plot function.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments

        """

        self._fmax = fmax
        self.relative = relative
        self.time_range = time_range
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
        raise ValueError("fmax must be specified when plotting absolut frequencies.")

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
                    t_min, dt, timeseries = binned_timeseries(
                        times, what=property, n=n, t_range=self.time_range
                    )

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

                    # cut data above fmax which was only added to increase FFT performance
                    visible = freq <= fmax
                    freq, mag = freq[visible], mag[visible]

                    # expression wrappers
                    mag = prop.evaluate_expression(mag)
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
        dt_max,
        bin_time=None,
        bin_count=None,
        exact_bin_time=True,
        log=True,
        mask=None,
        time_range=None,
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
            dt_max (float): Maximum interval (in s) to plot.
            bin_time (float): Time bin width (in s) if bin_count is None.
            bin_count (int): Number of bins if bin_time is None.
            exact_bin_time (bool): What to do if bin_time is given but dt_max is not an exact multiple of it.
                If True, dt_max is adjusted to be a multiple of bin_time.
                If False, bin_time is adjusted instead.
            log (bool): If True, plot on a log scale.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            time_range (tuple): Time range of particles to consider. If None, all particles are considered.
            plot_kwargs (dict): Keyword arguments passed to the plot function.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments


        """

        if bin_time is not None:
            if exact_bin_time:
                dt_max = bin_time * round(dt_max / bin_time)
            else:
                bin_time = dt_max / round(dt_max / bin_time)

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
        self.time_range = time_range
        self.dt_max = dt_max

        # create plot elements
        def create_artists(i, j, k, ax, p):
            return ax.step([], [], **defaults(plot_kwargs, lw=1))[0]

        self._create_artists(create_artists)

        # Format plot axes
        ax = self.axis(-1)
        ax.set(xlim=(self.bin_time if log else 0, self.dt_max * self.factor_for("t")))
        if log:
            ax.set(xscale="log", yscale="log")
        else:
            ax.set(ylim=(0, None))

        # set data
        if particles is not None:
            self.update(particles, mask=mask, autoscale=True)

    @property
    def bin_time(self):
        return self._bin_time or self.dt_max / self._bin_count

    @property
    def bin_count(self):
        return int(np.ceil(self.dt_max / self.bin_time))

    def update(self, particles, mask=None, autoscale=False):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (bool): Whether or not to perform autoscaling on all axes.
        """

        # extract times
        times = self._get_masked(particles, "t", mask)
        if self.time_range:
            times = times[(self.time_range[0] <= times) & (times < self.time_range[1])]
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

    @staticmethod
    def _link_cv_duty_axes(
        ax, at, twin_is_duty=True, orientation="y", factor_cv=1, factor_duty=1
    ):
        """Link twin axis with corresponding metric (cv or duty)

        This ties the limits of the twin axis to the primary axis (registering change listeners)
        and also applies the formatters and tick locators for coefficient-of-variation (cv) and
        duty-factor (duty) metrices.

        Args:
            ax (mpl.axis.Axis): The primary axis (the limits of which might change)
            at (mpl.axis.Axis): The twin axis
            twin_is_duty (bool): True if cv data is on `ax` and duty data is on `at`. False if that is swapped.
            orientation (str): "x" if data is on x-axis, "y" otherwise
            factor_cv (float): Optional factor associated with cv axis
            factor_duty (float): Optional factor associated with duty axis
        """
        xy = "x" if orientation[0].lower() in "xh" else "y"
        ax_cv, ax_duty = (ax, at) if twin_is_duty else (at, ax)

        cv2duty = lambda cv: factor_duty / (1 + (cv / factor_cv) ** 2)
        duty2cv = lambda du: factor_cv * (factor_duty / du - 1) ** 0.5

        # cv axis
        axis_cv = getattr(ax_cv, f"{xy}axis")
        ax_cv.set(**{f"{xy}label": MetricesMixin._metric_properties.get("cv").label_for_axes()})

        # duty axis
        axis_duty = getattr(ax_duty, f"{xy}axis")
        ax_duty.set(
            **{f"{xy}label": MetricesMixin._metric_properties.get("duty").label_for_axes()}
        )

        # tick locators and formatters
        if twin_is_duty:
            # duty ticks based on cv ticks
            granularity = 1 / 100 * factor_duty
            axis_duty.set_major_locator(
                TwinFunctionLocator(axis_cv.get_major_locator(), cv2duty, duty2cv, granularity)
            )
            axis_duty.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda cv, i: f"{100*cv2duty(cv):.0f} %")
            )
            axis_duty.set_minor_locator(
                mpl.ticker.FixedLocator(duty2cv(granularity * np.linspace(1, 100, 100)))
            )
            # ensure cv >= 0
            axis_cv.set_major_locator(TransformedLocator(axis_cv.get_major_locator(), vmin=0))

        else:
            # ct ticks based on duty ticks
            granularity = 0.1 * factor_cv
            axis_cv.set_major_locator(
                TwinFunctionLocator(axis_duty.get_major_locator(), duty2cv, cv2duty, granularity)
            )
            axis_cv.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda du, i: "∞" if du <= 0 else f"{duty2cv(du):.1f}")
            )
            axis_cv.set_minor_locator(
                TransformedLocator(
                    mpl.ticker.MultipleLocator(granularity),
                    duty2cv,
                    cv2duty,
                    vmin=0.05 * factor_duty,
                    vmax=factor_duty,
                )
            )
            # ensure 0 < duty <= 1
            axis_duty.set_major_locator(
                TransformedLocator(axis_duty.get_major_locator(), vmin=1e-9, vmax=factor_duty)
            )

        # link twin axis limits to primary axis
        at.set_navigate(False)  # we maintain limits ourselves
        ax.set_zorder(at.get_zorder() + 1)  # make sure axis on top to capture events
        if xy == "x":
            callback = lambda a: at.set(xlim=a.get_xlim())
        else:
            callback = lambda a: at.set(ylim=a.get_ylim())
        ax.callbacks.connect(f"{xy}lim_changed", callback)

        # force update once to initial values
        if xy == "x":
            ax.set(xlim=ax.get_xlim())
        else:
            ax.set(ylim=ax.get_ylim())

    def _format_metric_axes(self, add_compatible_twin_axes):
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                a = self.axis(i, j)
                if np.all(np.array(pp) == "duty"):
                    a.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

            # indicate compatible metric on opposite axis (if free space)
            if add_compatible_twin_axes and len(ppp) == 1:
                if np.all(np.array(ppp[0]) == "duty"):
                    twin_is_duty = False
                elif np.all(np.array(ppp[0]) == "cv"):
                    twin_is_duty = True
                else:
                    twin_is_duty = None

                if twin_is_duty is not None:
                    a = self.axis(i)
                    at = a.twinx()
                    self._link_cv_duty_axes(
                        a,
                        at,
                        twin_is_duty,
                        factor_duty=self.factor_for("duty"),
                        factor_cv=self.factor_for("cv"),
                    )
                    at.set(ylabel=self.label_for("duty" if twin_is_duty else "cv"))


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
        t_min, dt, counts = binned_timeseries(times, n=ncbins)
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
        counting_dt_min=None,
        counting_dt_max=None,
        counting_bins_per_evaluation=50,
        std=True,
        poisson=True,
        mask=None,
        time_range=None,
        log=True,
        plot_kwargs=None,
        ignore_insufficient_statistics=False,
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
            counting_dt_min (float): Minimum time bin width for counting.
            counting_dt_max (float): Maximum time bin width for counting.
            counting_bins_per_evaluation (int): Number of counting bins used to evaluate metric over.
                Use None to evaluate metric once over all bins. Otherwise, the metric is evaluated
                over each ``counting_bins_per_evaluation`` consecutive bins, and average and std of
                all evaluations plotted. This suppresses fluctuations on larger timescales to affect
                the metric of smaller timescales.
            std (bool): Whether or not to plot standard deviation of variability.
                Only relevant if counting_bins_per_evaluation is not None.
            poisson (bool): Whether or not to plot the Poisson limit.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            time_range (tuple): Time range of particles to consider. If None, all particles are considered.
            log (bool): Whether or not to plot the x-axis in log scale.
            plot_kwargs (dict): Keyword arguments passed to the plot function.
            ignore_insufficient_statistics (bool): When set to True, the plot will include data with insufficient statistics.
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
        std = std and self.counting_bins_per_evaluation

        # Format plot axes
        self._format_metric_axes(kwargs.get("ax") is None)
        for a in self.axflat:
            a.set(xscale="log" if self.log else "lin")

        # Create plot elements
        def create_artists(i, j, k, ax, p):
            kwargs = defaults(plot_kwargs, label=self._legend_label_for(p))
            plot = ax.plot([], [], **kwargs)[0]
            kwargs.update(color=plot.get_color())
            if std:
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
        if std:
            # merge plot and errorbar patches
            for i, h in enumerate(self._legend_entries):
                labels = [h[0].get_label()] + [_.get_label() for _ in h[2:]]
                self._legend_entries[i] = [tuple(h[0:2])] + h[2:]
                self.legend(i, show="auto", labels=labels)

        # set data
        if particles is not None:
            self.update(
                particles,
                mask=mask,
                autoscale=True,
                ignore_insufficient_statistics=ignore_insufficient_statistics,
            )

    def update(
        self, particles, mask=None, autoscale=False, *, ignore_insufficient_statistics=False
    ):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (bool): Whether or not to perform autoscaling on all axes.
            ignore_insufficient_statistics (bool): When set to True, the plot will include data with insufficient statistics.

        Returns:
            Changed artists
        """

        # extract times
        times = self._get_masked(particles, "t", mask)
        if self.time_range:
            times = times[(self.time_range[0] <= times) & (times < self.time_range[1])]
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
            )
            if not ignore_insufficient_statistics:
                print(f"Nothing plotted.")
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
            _, DT[i], N = binned_timeseries(times, n=nbin)

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


class TimeBinMetricHelper(ParticlePlotMixin, MetricesMixin):
    def __init__(self, *, twiss=None, beta=None, frev=None, circumference=None):
        """Helper class for binning and evaluating metrices on timeseries data.

        Args:
            twiss (dict | None): Twiss parameters (alfx, alfy, betx and bety) to use for conversion to normalized phase space coordinates.
            beta (float | None): Relativistic beta of particles. Defaults to particles.beta0.
            frev (float | None): Revolution frequency of circular line for calculation of particle time.
            circumference (float | None): Path length of circular line if frev is not given.

        """
        self._init_particle_mixin(twiss=twiss, beta=beta, frev=frev, circumference=circumference)

    def binned_timeseries(
        self, particles, dt=None, *, mask=None, t_range=None, what=None, moments=1
    ):
        """Get binned timeseries with equally spaced time bins

        Args:
            particles (Any): Particles data to plot.
            dt (float | None): Bin width in seconds.
            mask (None | Any | callable): The mask. Can be None, a slice, a binary mask or a callback.
                If a callback, it must have the signature ``(mask_1, get) -> mask_2`` where mask_1 is the
                binary mask to be modified, mask_2 is the modified mask, and get is a method allowing the
                callback to retriev particle properties in their respective data units.
                Example:
                    def mask_callback(mask, get):
                        mask &= get("t") < 1e-3  # all particles with time < 1 ms
                        return mask
            t_range (tuple[float] | None): Tuple of (min, max) time values to consider. If None, the range is determined from the data.
            what (str | None): Property to return per bin. Defaults to None, i.e. return counts per bins.
            moments (int | list[int]): The moment(s) to return if what is not None.
                Allows to get the mean (1st moment, default), variance (difference between 2nd and 1st moment) etc.

        Returns:
            tuple[np.array]: Tuple of (t_min, dt_count, values) where
                t_min is the time of the first bin,
                dt_count is the bin width and
                values is the array of counts per bin (or whatever `what` was set to).
        """
        # extract times
        times = self._get_masked(particles, "t", mask)
        data = None if what is None else self._get_masked(particles, what, mask)

        # bin into counting bins
        t_min, dt_count, values = binned_timeseries(
            times, what=data, dt=dt, t_range=t_range, moments=moments
        )

        return t_min, dt_count, values

    def calculate_metric(self, counts, metric, nbins):
        """Calculate metric on timeseries

        Args:
            counts (np.array): 1D timeseries of counts per bin.
            metric (str): Metric to calculate. See :class:`MetricesMixin` for available metrics.
            nbins (int): Number of subsequent bins to evaluate metric over.

        Returns:
            tuple[np.array]: Tuple of (value, limit) arrays for each evaluation of the metric.
        """
        # make 2D array by subdividing into evaluation bins
        N = counts[: int(len(counts) / nbins) * nbins].reshape((-1, nbins))

        # calculate metrics
        F, F_limit = self._calculate_metric(N, metric, axis=1)

        return F, F_limit


## Restrict star imports to local namespace
__all__ = [
    name
    for name, thing in globals().items()
    if not (name.startswith("_") or isinstance(thing, types.ModuleType))
]
