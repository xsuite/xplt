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

from .util import *
from .base import XManifoldPlot, TwinFunctionLocator, TransformedLocator
from .particles import ParticlePlotMixin, ParticlesPlot, ParticleHistogramPlotMixin
from .properties import Property, DerivedProperty, arb_unit


def binned_timeseries(
    times, *, what=None, n=None, dt=None, t_range=None, moments=1, make_n_power_of_two=False
):
    """

    .. deprecated:: 0.7
        Use :func:`xplt.util.binned_data` instead.
    """
    return binned_data(
        times,
        what=what,
        n=n,
        dv=dt,
        v_range=t_range,
        moments=moments,
        make_n_power_of_two=make_n_power_of_two,
    )


class TimePlot(ParticlesPlot):
    """A plot of particle properties as function of time"""

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


class TimeBinPlot(XManifoldPlot, ParticlePlotMixin, ParticleHistogramPlotMixin):
    """A binned histogram plot of particles as function of times"""

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
        time_offset=0,
        plot_kwargs=None,
        **kwargs,
    ):
        """

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
                If `kind` is a particle property, this has no effect.
            moment (int): The moment(s) to plot if kind is a particle property.
                Allows to get the mean (1st moment, default), variance (difference between 2nd and 1st moment) etc.

            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            time_range (tuple): Time range of particles to consider. If None, all particles are considered.
            time_offset (float): Time offset for x-axis is seconds, i.e. show values as `t-time_offset`.
            plot_kwargs (dict): Keyword arguments passed to the plot function, see :meth:`matplotlib.axes.Axes.plot`.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments

        """
        self.moment = moment
        kwargs = self._init_particle_mixin(**kwargs)
        kwargs = self._init_particle_histogram_mixin(**kwargs)
        kwargs["_properties"] = defaults(
            kwargs.get("_properties"),
            t_offset=DerivedProperty("$t-t_0$", "s", lambda t: t - time_offset),
        )
        kwargs["display_units"] = defaults(
            kwargs.get("display_units"),
            t_offset=kwargs.get("display_units", {}).get("t"),
        )

        super().__init__(on_x="t_offset" if time_offset else "t", on_y=kind, **kwargs)

        if bin_time is None and bin_count is None:
            bin_count = 100
        if bin_time is not None and bin_count is not None:
            raise ValueError("Only one of bin_time or bin_count may be specified.")
        self.bin_time = bin_time
        self.bin_count = bin_count
        self.relative = relative
        self.time_range = time_range

        # Format plot axes
        self.axis(-1).set(xlabel=self.label_for(self.on_x), ylim=(0, None))
        if self.relative:
            for a in self.axflat:
                a.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

        # Create plot elements
        def create_artists(i, j, k, ax, p):
            kwargs = defaults_for(
                "plot", plot_kwargs, lw=1, label=self._legend_label_for((i, j, k))
            )
            if self._count_based(p):
                kwargs = defaults_for("plot", kwargs, drawstyle="steps-pre")
            return ax.plot([], [], **kwargs)[0]

        self._create_artists(create_artists)

        # set data
        if particles is not None:
            self.update(particles, mask=mask, autoscale=True)

    def _symbol_for(self, p):
        symbol = super()._symbol_for(p)
        if p != self.on_x and self.moment is not None and not self._count_based(p):
            # it is averaged
            symbol = f"$\\langle${symbol}$\\rangle$"
        return symbol

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
        t_prop = self.prop(self.on_x)
        times = t_prop.values(particles, mask, unit="s")

        # update plots
        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                count_based = False
                edges = None
                for k, p in enumerate(pp):
                    count_based = self._count_based(p)
                    if p in ("current", "charge"):
                        property = self.prop("q").values(particles, mask)
                    elif count_based:
                        property = None
                    else:
                        property = self.prop(p).values(particles, mask)

                    t_min, dt, timeseries = binned_data(
                        times,
                        what=property,
                        n=self.bin_count,
                        dv=self.bin_time,
                        v_range=self.time_range,
                        moments=None if count_based else self.moment,
                    )
                    timeseries = timeseries.astype(np.float64)
                    edges = np.linspace(t_min, t_min + dt * timeseries.size, timeseries.size + 1)

                    self.annotate(
                        f'$\\Delta t_\\mathrm{{bin}} = {pint.Quantity(dt, "s"):#~.4gL}$'
                    )

                    if self.relative:
                        if not count_based:
                            raise ValueError(
                                "Relative plots are only supported for kind 'count', 'rate', 'cumulative', 'charge' or 'current'."
                            )
                        timeseries /= len(times)

                    if p in ("rate", "current"):
                        timeseries /= dt

                    # post-processing expression wrappers
                    if wrap := self.on_y_expression[i][j][k]:
                        timeseries = evaluate_expression_wrapper(wrap, p, timeseries)

                    # display units
                    edges *= self.factor_for(self.on_x)
                    timeseries *= self.factor_for(p)

                    # update plot
                    if p == "cumulative":
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
                    if self.time_range is not None and edges is not None:
                        a.set(xlim=(min(edges), max(edges)))

        return changed


class TimeFFTPlot(XManifoldPlot, ParticlePlotMixin, ParticleHistogramPlotMixin):
    """A frequency plot based on particle arrival times"""

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
        timeseries=None,
        timeseries_fs=None,
        time_range=None,
        plot_kwargs=None,
        **kwargs,
    ):
        """

        The particle arrival time is:
            - For circular lines: at_turn / frev - zeta / beta / c0
            - For linear lines: zeta / beta / c0

        From the particle arrival times (non-equally distributed timestamps), a timeseries with equally
        spaced time bins is derived. The time bin size is determined based on fmax and performance considerations.
        By default, the binned property is the number of particles arriving within the bin time (what='count').
        Alternatively, a particle property may be specified (e.g. what='x'), in which case that property is
        averaged over all particles arriving in the respective bin. The FFT is then computed over the timeseries.

        Useful to plot time structures of particles loss, such as spill structures.

        Instead of particle timestamps, it is also possible to pass already binned timeseries data to the plot.

        Args:
            particles (Any): Particles data to plot.
            kind (str | list): Defines the properties to make the FFT over, including 'count' (default), or a particle property to average.
                This is a manifold subplot specification string like ``"count-cumulative"``, see :class:`~.base.XManifoldPlot` for details.
                In addition, abbreviations for x-y-parameter pairs are supported (e.g. ``P`` for ``Px+Py``).
            fmax (float): Maximum frequency (in Hz) to plot.
            relative (bool): If True, plot relative frequencies (f/frev) instead of absolute frequencies (f).
            log (bool): If True, plot on a log scale.
            scaling (str | dict): Scaling of the FFT. Can be 'amplitude', 'pds' or 'pdspp' or a dict with a scaling per property where
                                  `amplitude` (default for non-count based properties) scales the FFT magnitude to the amplitude,
                                  `pds` (power density spectrum, default for count based properties) scales the FFT magnitude to power,
                                  `pdspp` (power density spectrum per particle) is simmilar to 'pds' but normalized to particle number.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            timeseries (dict[str, np.array]): Pre-binned timeseries data as alternative to timestamp-based particle data.
                                              The dictionary must contain keys for each `kind` (e.g. `count`).
                                              When specified, `timeseries_fs` must also be set, and `particles` and `mask` must be None.
            timeseries_fs (float): The sampling frequency for the timeseries data.
            time_range (tuple): Time range of particles to consider. If None, all particles are considered.
            plot_kwargs (dict): Keyword arguments passed to the plot function, see :meth:`matplotlib.axes.Axes.plot`.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments

        """

        self._fmax = fmax
        self.relative = relative
        self.time_range = time_range
        self._scaling = scaling
        if log is None:
            log = not relative

        kwargs = self._init_particle_mixin(**kwargs)
        kwargs = self._init_particle_histogram_mixin(**kwargs)
        kwargs["_properties"] = defaults(
            kwargs.get("_properties"),
            f=Property("$f$", "Hz", description="Frequency"),
        )
        super().__init__(on_x=None, on_y=kind, **kwargs)  # handled manually

        # Format plot axes
        self.axis(-1).set(xlabel="$f/f_{rev}$" if self.relative else self.label_for("f"))
        for a in self.axflat:
            a.set(ylim=(0, None))
            if log:
                a.set(xscale="log", yscale="log")

        # Create plot elements
        def create_artists(i, j, k, ax, p):
            kwargs = defaults_for(
                "plot", plot_kwargs, lw=1, label=self._legend_label_for((i, j, k))
            )
            return ax.plot([], [], **kwargs)[0]

        self._create_artists(create_artists)

        # set data
        if particles is not None or timeseries is not None:
            self.update(
                particles,
                mask=mask,
                timeseries=timeseries,
                timeseries_fs=timeseries_fs,
                autoscale=True,
            )

    def _get_scaling(self, key):
        if isinstance(self._scaling, str):
            return self._scaling.lower()
        if isinstance(self._scaling, dict) and key in self._scaling:
            return self._scaling[key].lower()
        return "pds" if self._count_based(key) else "amplitude"

    def fmax(self, particles=None, *, default=None):
        """Return the maximum frequency this plot should show

        Args:
            particles (Any): Particle data for determination of revolution frequency
            default (float | None): Default value to return if maximum frequency can not be determined

        Returns:
             float: maximum frequency

        Raises:
            ValueError: If maximum frequency can not be determined and no default was provided
        """
        if self._fmax is not None:
            return self._fmax
        if self.relative:
            return self.frev(particles)
        if default is not None:
            return default
        raise ValueError("fmax must be specified when plotting absolut frequencies.")

    def update(
        self, particles=None, mask=None, autoscale=False, *, timeseries=None, timeseries_fs=None
    ):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (bool): Whether to perform autoscaling on all axes.
            timeseries (dict[str, np.array]): Pre-binned timeseries data as alternative to timestamp-based particle data.
                                              The dictionary must contain keys for each `kind` (e.g. `count`).
                                              When specified, `timeseries_fs` must also be set, and `particles` and `mask` must be None.
            timeseries_fs (float | dict[str, float]): The sampling frequency for the timeseries data, or a dictionary with
                                                      sampling frequencies for each entry in `timeseries`.

        Returns:
            list: Changed artists
        """

        if particles is not None:
            if timeseries is not None or timeseries_fs is not None:
                raise ValueError(
                    "`timeseries` and `timeseries_fs` must be None when passing data via `particles`"
                )

            # extract times
            times = self.prop("t").values(particles, mask, unit="s")
            fmax = self.fmax(particles)
            ppscale = len(times)

            # compute binned timeseries
            timeseries, timeseries_fs = {}, {}
            for p in self.on_y_unique:
                prop = self.prop(p)
                property = None if self._count_based(p) else prop.values(particles, mask)
                # to improve FFT performance, round up to next power of 2
                _, dt, ts = binned_data(
                    times,
                    what=property,
                    dv=1 / (2 * fmax),
                    v_range=self.time_range,
                    make_n_power_of_two=True,
                )
                timeseries[p] = ts
                timeseries_fs[p] = 1 / dt

        elif timeseries is not None:
            if particles is not None or mask is not None:
                raise ValueError(
                    "`particles` and `mask` must be None when passing data via `timeseries`"
                )
            if timeseries_fs is None:
                raise ValueError("`timeseries_fs` is required when passing data via `timeseries`")

            if not isinstance(timeseries_fs, dict):
                timeseries_fs = {p: timeseries_fs for p in self.on_y_unique}

            # binned timeseries provided by user, apply time range
            fmax = np.nan
            ppscale = np.sum(get(timeseries, "count", [1]))
            for p in timeseries:
                fs, ts = timeseries_fs[p], np.array(timeseries[p])
                fmax = np.nanmax([fmax, self.fmax(default=fs)])
                if self.time_range is not None:
                    i_min, i_max = [None if t is None else int(t * fs) for t in self.time_range]
                    timeseries[p] = ts[i_min:i_max]

        else:
            raise ValueError("Data was neither passed via `particles` nor `timeseries`")

        # update plots
        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):

                    # calculate fft without DC component
                    dt, ts = 1 / get(timeseries_fs, p), get(timeseries, p)
                    freq = np.fft.rfftfreq(ts.size, d=dt)[1:]
                    mag = np.abs(np.fft.rfft(ts))[1:]

                    # scale frequency according to user preferences
                    if self.relative:
                        freq *= 1 / self.frev(particles)
                    else:
                        freq *= self.factor_for("f")

                    # scale magnitude according to user preference
                    if p == "rate":
                        mag /= dt
                    if self._get_scaling(p) == "amplitude":
                        # amplitude in units of p
                        mag *= 2 / len(ts) * self.factor_for(p)
                    elif self._get_scaling(p) in ("pds", "pdspp"):
                        # power density spectrum in arb. unit
                        mag = mag**2
                        if self._get_scaling(p) == "pdspp":
                            mag /= ppscale  # per particle

                    # cut data above fmax which was only added to increase FFT performance
                    visible = freq <= fmax
                    freq, mag = freq[visible], mag[visible]

                    # post-processing expression wrappers
                    if wrap := self.on_y_expression[i][j][k]:
                        mag = evaluate_expression_wrapper(wrap, p, mag)

                    if p == "cumulative":
                        mag = np.cumsum(mag)

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

        self.annotate(f"$\\Delta t_\\mathrm{{bin}} = {pint.Quantity(dt, 's'):#~.4gL}$")

        return changed

    def _symbol_for(self, p):
        symbol = super()._symbol_for(p)
        if p not in "f":
            # it is the FFT of it
            symbol = symbol.strip("$")
            if self._get_scaling(p) == "amplitude":
                symbol = f"$\\hat{{{symbol}}}$"
            elif self._get_scaling(p) == "pds":
                symbol = f"$|\\mathrm{{FFT({symbol})}}|^2$"
            elif self._get_scaling(p) == "pdspp":
                symbol = f"$|\\mathrm{{FFT({symbol})}}|^2/N_\\mathrm{{total}}$"
            else:
                symbol = f"$|\\mathrm{{FFT({symbol})}}|$"
        return symbol

    def display_unit_for(self, p):
        if p not in "f" and self._get_scaling(p) != "amplitude":
            return arb_unit
        return super().display_unit_for(p)

    def plot_harmonics(self, f, df=0, *, n=20, relative=False, **plot_kwargs):
        """Add vertical lines or spans indicating the location of values or spans and their harmonics

        Args:
            f (float | list[float] | np.array): Fundamental frequency or list of frequencies.
            df (float | list[float] | np.array, optional): Bandwidth or list of bandwidths centered around frequencies(s) in Hz.
            n (int): Number of harmonics to plot.
            relative (bool): If true, then `f` and `df` are interpreted as relative frequencies (f/frev).
                             Otherwise they are interpreted as absolute frequencies in Hz (default).
            plot_kwargs: Keyword arguments to be passed to plotting method
        """
        s = 1
        if relative and not self.relative:  # convert from relative to absolute
            s = self.frev()
        elif not relative and self.relative:  # convert from absolute to relative
            s = 1 / self.frev()

        if not self.relative:  # unit conversion
            s *= self.factor_for("f")

        f, df = s * np.array(f, ndmin=1), s * np.array(df, ndmin=1)
        for a in self.axflat:
            super().plot_harmonics(a, f, df, n=n, **plot_kwargs)


class TimeIntervalPlot(XManifoldPlot, ParticlePlotMixin, ParticleHistogramPlotMixin):
    """A histogram plot of particle arrival intervals (i.e. delay between consecutive particles)"""

    def __init__(
        self,
        particles=None,
        kind="count",
        *,
        dt_max,
        bin_time=None,
        bin_count=None,
        exact_bin_time=True,
        relative=False,
        log=True,
        poisson=False,
        mask=None,
        time_range=None,
        plot_kwargs=None,
        poisson_kwargs=None,
        **kwargs,
    ):
        """

        The plot is based on the particle arrival time, which is:
            - For circular lines: at_turn / frev - zeta / beta / c0
            - For linear lines: zeta / beta / c0

        Useful to plot time structures of particles loss, such as spill structures.

        Args:
            particles (Any): Particles data to plot.
            kind (str | list): Defines the properties to plot, including 'count' (default), 'rate' or 'cumulative'.
                This is a manifold subplot specification string like ``"count,cumulative"``, see :class:`~.base.XManifoldPlot` for details.
            dt_max (float): Maximum interval (in s) to plot.
            bin_time (float): Time bin width (in s) if bin_count is None.
            bin_count (int): Number of bins if bin_time is None.
            exact_bin_time (bool): What to do if bin_time is given but dt_max is not an exact multiple of it.
                If True, dt_max is adjusted to be a multiple of bin_time.
                If False, bin_time is adjusted instead.
            relative (bool): If True, plot relative numbers normalized to total count.
                If `kind` is a particle property, this has no effect.
            log (bool | str): To make the plot log scaled, can be any of (True, 'x', 'y', 'xy' or False).
            poisson (bool): If true, indicate ideal poisson distribution.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            time_range (tuple): Time range of particles to consider. If None, all particles are considered.
            plot_kwargs (dict): Keyword arguments passed to the plot function, see :meth:`matplotlib.axes.Axes.plot`.
            poisson_kwargs (dict): Additional keyword arguments passed to the plot function for Poisson limit.
                                   See :meth:`matplotlib.axes.Axes.plot` (only applicable if `poisson` is True).
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments


        """

        if bin_time is not None:
            if exact_bin_time:
                dt_max = bin_time * round(dt_max / bin_time)
            else:
                bin_time = dt_max / round(dt_max / bin_time)

        kwargs = self._init_particle_mixin(**kwargs)
        kwargs = self._init_particle_histogram_mixin(**kwargs)
        kwargs["_properties"] = defaults(
            kwargs.get("_properties"),
            dt=Property("$\\Delta t$", "s", description="Delay between consecutive particles"),
        )
        super().__init__(on_x="dt", on_y=kind, **kwargs)

        if bin_time is None and bin_count is None:
            bin_count = 100
        self._bin_time = bin_time
        self._bin_count = bin_count
        self.relative = relative
        self.time_range = time_range
        self.dt_max = dt_max

        # Create plot elements
        def create_artists(i, j, k, ax, p):
            kwargs = defaults_for(
                "plot", plot_kwargs, lw=1, label=self._legend_label_for((i, j, k))
            )
            if self._count_based(p):
                kwargs = defaults_for("plot", kwargs, drawstyle="steps-pre")
            else:
                raise ValueError(f"Property `{p}` not supported")
            plot = ax.plot([], [], **kwargs)[0]
            if poisson:
                kwargs.update(
                    defaults_for(
                        "plot",
                        poisson_kwargs,
                        color=plot.get_color() or "gray",
                        alpha=0.5,
                        zorder=1.9,
                        lw=1,
                        ls=":",
                        label="Poisson ideal",
                    )
                )
                pplot = ax.plot([], [], **kwargs)[0]
            else:
                pplot = None
            return plot, pplot

        self._create_artists(create_artists)

        # Format plot axes
        self.axis(-1).set(xlim=(self.bin_time if log else 0, self.dt_max * self.factor_for("t")))
        for a in self.axflat:
            if log in (True, "x", "xy"):
                a.set(xscale="log")
            if log in (True, "y", "xy"):
                a.set(yscale="log")
            else:
                a.set(ylim=(0, None))
            if self.relative:
                a.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

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
        times = self.prop("t").values(particles, mask, unit="s")
        if self.time_range:
            times = times[(self.time_range[0] <= times) & (times < self.time_range[1])]
        delay = self.factor_for("t") * np.diff(sorted(times))

        self.annotate(f"$\\Delta t_\\mathrm{{bin}} = {pint.Quantity(self.bin_time, 's'):#~.4gL}$")

        # update plots
        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                edges = None
                for k, p in enumerate(pp):
                    if p in ("current", "charge"):
                        weights = self.prop("q").values(particles, mask)
                    elif self._count_based(p):
                        weights = None
                    else:
                        raise NotImplementedError()

                    # calculate histogram
                    counts, edges = np.histogram(
                        delay,
                        bins=self.bin_count,
                        range=(0, self.bin_count * self.bin_time),
                        weights=weights,
                    )
                    counts = counts.astype(np.float)
                    if p in ("rate", "current"):
                        counts /= self.bin_time
                    if self.relative:
                        counts /= len(delay)
                    counts *= self.factor_for(p)

                    # update plot
                    plot, pplot = self.artists[i][j][k]
                    if p == "cumulative":
                        # steps open after last bin
                        counts = np.concatenate(([0], np.cumsum(counts)))
                    elif self._count_based(p):
                        # steps go back to zero after last bin
                        edges = np.append(edges, edges[-1])
                        counts = np.concatenate(([0], counts, [0]))
                    else:
                        raise NotImplementedError()
                    plot.set_data((edges, counts))
                    changed.append(plot)

                    # poisson limit
                    if pplot:
                        rate = times.size / (np.max(times) - np.min(times))
                        if p == "cumulative":
                            limit = 1 - np.exp(-rate * edges)
                            limit *= np.max(counts)
                        else:
                            limit = rate * np.exp(-rate * edges)
                            limit *= np.sum(counts) * self.bin_time
                        pplot.set_data((edges, limit))
                        changed.append(pplot)

                if autoscale:
                    ax = self.axis(i, j)
                    ax.relim()
                    ax.autoscale()
                    if not ax.get_yscale() == "log":
                        ax.set(ylim=(0, None))
                    if edges is not None:
                        e = edges[edges > 0] if ax.get_xscale() == "log" else edges
                        if e.size > 1:
                            ax.set(xlim=(min(e), max(e)))

        return changed

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
        cv=Property("$c_v=\\sigma/\\mu$", "1", description="Coefficient of variation"),
        duty=Property(
            "$F=\\langle N \\rangle^2/\\langle N^2 \\rangle$",
            "1",
            description="Spill duty factor",
        ),
        maxmean=Property(
            "$M=\\hat{N}/\\langle N \\rangle$", "1", description="Max-to-mean ratio"
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
        prop_cv = MetricesMixin._metric_properties.get("cv")
        ax_cv.set(**{f"{xy}label": f"{prop_cv.description or ''}   {prop_cv.symbol}".strip()})

        # duty axis
        axis_duty = getattr(ax_duty, f"{xy}axis")
        prop_duty = MetricesMixin._metric_properties.get("duty")
        ax_duty.set(
            **{f"{xy}label": f"{prop_duty.description or ''}   {prop_duty.symbol}".strip()}
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
    """Plot variability of particle time on microscopic scale as function of time on macroscopic scale"""

    def __init__(
        self,
        particles=None,
        kind="cv",
        *,
        counting_dt=None,
        counting_bins=None,
        evaluate_dt=None,
        evaluate_bins=None,
        poisson=True,
        mask=None,
        time_range=None,
        time_offset=0,
        plot_kwargs=None,
        poisson_kwargs=None,
        **kwargs,
    ):
        """

        The particle arrival times are histogramed into counting bins, the width of which
        corresponds to the time resolution of a detector (``counting_dt``).
        The plot estimates fluctuations in these particle counts by applying a metric
        over an evaluation window (``evaluation_dt``).

        See :class:`~.timestructure.MetricesMixin` for a list of implemented metrics.

        If the particle data corresponds to particles lost at the extraction septum,
        the plot yields the spill quality as function of extraction time.

        Args:
            particles (Any): Particles data to plot.
            kind (str | list): Metric to plot. See above for list of implemented metrics.
            counting_dt (float): Time bin width for counting if counting_bins is None.
            counting_bins (int): Number of bins if counting_dt is None.
            evaluate_dt (float): Time bin width for metric evaluation if evaluate_bins is None.
            evaluate_bins (int): Number of bins if evaluate_dt is None.
            poisson (bool): If true, indicate poisson limit.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            time_range (tuple): Time range of particles to consider. If None, all particles are considered.
            time_offset (float): Time offset for x-axis is seconds, i.e. show values as `t-time_offset`.
            plot_kwargs (dict): Keyword arguments passed to the plot function, see :meth:`matplotlib.axes.Axes.step`.
            poisson_kwargs (dict): Additional keyword arguments passed to the plot function for Poisson limit.
                                   See :meth:`matplotlib.axes.Axes.step` (only applicable if `poisson` is True).
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments

        """
        kwargs = self._init_particle_mixin(**kwargs)
        kwargs["_properties"] = defaults(
            kwargs.get("_properties"),
            t_offset=DerivedProperty("$t-t_0$", "s", lambda t: t - time_offset),
            **self._metric_properties,
        )
        kwargs["display_units"] = defaults(
            kwargs.get("display_units"), t_offset=kwargs.get("display_units", {}).get("t")
        )
        super().__init__(on_x="t_offset" if time_offset else "t", on_y=kind, **kwargs)

        if counting_dt is None and counting_bins is None:
            counting_bins = 100 * 100
        if evaluate_dt is None and evaluate_bins is None:
            evaluate_bins = 100
        self.counting_dt = counting_dt
        self.counting_bins = counting_bins
        self.evaluate_dt = evaluate_dt
        self.evaluate_bins = evaluate_bins
        self.time_range = time_range

        # Format plot axes
        self._format_metric_axes(kwargs.get("ax") is None)

        # Create plot elements
        def create_artists(i, j, k, ax, p):
            kwargs = defaults_for(
                "plot", plot_kwargs, lw=1, label=self._legend_label_for((i, j, k))
            )
            step = ax.step([], [], **kwargs)[0]
            if poisson:
                kwargs.update(
                    defaults_for(
                        "plot",
                        poisson_kwargs,
                        color=step.get_color() or "gray",
                        alpha=0.5,
                        zorder=1.9,
                        lw=1,
                        ls=":",
                        label="Poisson limit",
                    )
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
        t_prop = self.prop(self.on_x)
        times = t_prop.values(particles, mask, unit="s")

        # re-sample times into equally binned time series
        ncbins = self.counting_bins or int(
            np.ceil((np.max(times) - np.min(times)) / self.counting_dt)
        )
        if self.evaluate_bins is not None:
            nebins = int(ncbins / self.evaluate_bins)
        else:
            nebins = int(ncbins * self.evaluate_dt / (np.max(times) - np.min(times)))

        # bin into counting bins
        t_min, dt, counts = binned_data(times, n=ncbins, v_range=self.time_range)
        edges = np.linspace(t_min, t_min + dt * ncbins, ncbins + 1)

        # make 2D array by subdividing into evaluation bins
        N = counts[: int(len(counts) / nebins) * nebins].reshape((-1, nebins))
        E = edges[: int(len(edges) / nebins + 1) * nebins : nebins]

        self.annotate(
            f'$\\Delta t_\\mathrm{{count}} = {pint.Quantity(dt, "s"):#~.4gL}$\n'
            f'$\\Delta t_\\mathrm{{evaluate}} = {pint.Quantity(dt*nebins, "s"):#~.4gL}$'
        )

        # display units
        edges = np.append(E, E[-1])
        edges *= self.factor_for(self.on_x)

        # update plots
        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):
                    # calculate metrics
                    F, F_poisson = self._calculate_metric(N, p, axis=1)

                    # update plot
                    step, pstep = self.artists[i][j][k]
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
    """Plot variability of particle time as function of timescale"""

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
        std_kwargs=None,
        poisson_kwargs=None,
        ignore_insufficient_statistics=False,
        **kwargs,
    ):
        """

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
            std (bool): Whether or not to plot standard deviation of variability as errorbar.
                Only relevant if counting_bins_per_evaluation is not None.
            poisson (bool): Whether or not to plot the Poisson limit.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            time_range (tuple): Time range of particles to consider. If None, all particles are considered.
            log (bool): Whether or not to plot the x-axis in log scale.
            plot_kwargs (dict): Keyword arguments passed to the plot function. See :meth:`matplotlib.axes.Axes.plot`.
            std_kwargs (dict): Additional keyword arguments passed to the plot function for std errorbar.
                               See :meth:`matplotlib.axes.Axes.fill_between` (only applicable if `std` is True).
            poisson_kwargs (dict): Additional keyword arguments passed to the plot function for Poisson limit.
                                   See :meth:`matplotlib.axes.Axes.plot` (only applicable if `poisson` is True).
            ignore_insufficient_statistics (bool): When set to True, the plot will include data with insufficient statistics.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments


        """
        kwargs = self._init_particle_mixin(**kwargs)
        kwargs["_properties"] = defaults(
            kwargs.get("_properties"),
            tbin=Property("$\\Delta t_\\mathrm{count}$", "s", description="Time resolution"),
            **self._metric_properties,
        )
        super().__init__(on_x="tbin", on_y=kind, **kwargs)

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
            kwargs = defaults_for("plot", plot_kwargs, label=self._legend_label_for((i, j, k)))
            plot = ax.plot([], [], **kwargs)[0]
            kwargs.update(color=plot.get_color())
            if std:
                self._errkw = kwargs.copy()
                self._errkw.update(
                    defaults_for("fill_between", std_kwargs, zorder=1.8, alpha=0.3, ls="-", lw=0)
                )
                errorbar = ax.fill_between([], [], [], **self._errkw)
                errorbar._join_legend_entry_with = plot
            else:
                errorbar = None
            if poisson:
                kwargs.update(
                    defaults_for(
                        "plot", poisson_kwargs, zorder=1.9, ls=":", label="Poisson limit"
                    )
                )
                pstep = ax.plot([], [], **kwargs)[0]
            else:
                pstep = None
            return [plot, errorbar, pstep]

        self._create_artists(create_artists)

        # legend with combined patch
        if std:
            self.legend()

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
        times = self.prop("t").values(particles, mask, unit="s")
        if self.time_range:
            times = times[(self.time_range[0] <= times) & (times < self.time_range[1])]
        ntotal = times.size
        duration = np.max(times) - np.min(times)

        # annotate plot
        # f'$\\langle\\dot{{N}}\\rangle = {pint.Quantity(ntotal/duration, "1/s"):#~.4gL}$\n'
        self.annotate(
            "$\\Delta t_\\mathrm{evaluate} = "
            + (
                f"{self.counting_bins_per_evaluation:g}\\,\\Delta t_\\mathrm{{count}}$"
                if self.counting_bins_per_evaluation
                else f"{pint.Quantity(duration, 's'):#~.4gL}$"
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
            _, DT[i], N = binned_data(times, n=nbin)

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
                        join_legend_entry_with = errorbar._join_legend_entry_with
                        changed.append(errorbar)
                        errorbar.remove()
                        errorbar = ax.fill_between(
                            DT, F[p] - F_std[p], F[p] + F_std[p], **self._errkw
                        )
                        errorbar._join_legend_entry_with = join_legend_entry_with
                        self.artists[i][j][k][1] = errorbar
                        changed.append(errorbar)
                    if pstep:
                        pstep.set_data((DT, F_poisson[p]))
                        changed.append(pstep)

                if autoscale:
                    self._autoscale(ax, self.artists[i][j], tight="x")
                    ax.set(ylim=(0, None))

        return changed


class TimeBinMetricHelper(ParticlePlotMixin, MetricesMixin):
    """Helper class for binning and evaluating metrices on timeseries data"""

    def __init__(self, *, twiss=None, beta=None, frev=None, circumference=None):
        """

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
        times = self.get_property("t").values(particles, mask=mask, unit="s")
        data = what and self.get_property(what).values(particles, mask=mask)

        # bin into counting bins
        t_min, dt_count, values = binned_data(
            times, what=data, dv=dt, v_range=t_range, moments=moments
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
