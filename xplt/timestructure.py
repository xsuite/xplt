#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting particle arrival times

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-24"


from dataclasses import dataclass
import scipy.signal
import pint
from .util import *
from .base import XManifoldPlot, TwinFunctionLocator, TransformedLocator
from .particles import ParticlePlotMixin, ParticlesPlot, ParticleHistogramPlotMixin
from .properties import Property, DerivedProperty, arb_unit


def fmt(t, unit="s"):
    return f"{pint.Quantity(t, unit):#~.4gL}"


class _TimeBasePlot(XManifoldPlot):
    def __init__(
        self,
        *,
        time_range=None,
        time_offset=0.0,
        **kwargs,
    ):
        kwargs["_properties"] = defaults(
            kwargs.get("_properties"),
            t_offset=DerivedProperty("$t-t_0$", "s", lambda t: t - time_offset),
        )
        kwargs["display_units"] = defaults(
            kwargs.get("display_units"),
            t_offset=kwargs.get("display_units", {}).get("t"),
        )
        if "on_x" not in kwargs:
            kwargs["on_x"] = "t_offset" if time_offset else "t"

        super().__init__(**kwargs)

        self.time_range = time_range

    def _ensure_particles_timeseries_data(self, particles, timeseries):
        if particles is None and timeseries is None:
            raise ValueError("Data was neither passed via `particles` nor `timeseries`")
        elif particles is not None:
            if timeseries is not None:
                raise ValueError("`timeseries` must be None when passing data via `particles`")
        elif timeseries is not None:
            if particles is not None:
                raise ValueError("`particles` must be None when passing data via `timeseries`")
            if not isinstance(timeseries, dict):
                if len(self.on_y_unique) != 1:
                    raise ValueError("timeseries must be a dict of the form {kind: array}")
                timeseries = {self.on_y_unique[0]: timeseries}
            for ts in timeseries.values():
                if not isinstance(ts, Timeseries):
                    raise ValueError(
                        f"timeseries data must be of type ´xplt.Timeseries(waveform, dt=1/fs)´, found {type(ts)}"
                    )
        return timeseries

    def _get_times_in_range(self, particles, mask=None):
        """Cut the time data to time_range"""
        times = self.prop("t").values(particles, mask, unit="s")
        if self.time_range is not None and self.time_range[0] is not None:
            times = times[times >= self.time_range[0]]
        if self.time_range is not None and self.time_range[1] is not None:
            times = times[times < self.time_range[1]]
        return times


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
        if getattr(ax, f"get_{xy}scale")() != "linear":
            raise NotImplementedError(
                "Linked cv and duty axes are only supported for linear scaling!"
            )

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


PUBLIC_SECTION_BEGIN()


@dataclass
class Timeseries:
    """Class holding timeseries data

    Args:
        data: The timeseries waveform data array
        dt: The sampling period (in s) of the data
        t0: The time (in s) of the first data point
    """

    data: np.array
    dt: float
    t0: float = 0

    @property
    def fs(self):
        """Sampling frequency"""
        return 1 / self.dt

    @property
    def size(self):
        return self.data.size

    @staticmethod
    def from_timestamps(
        times, *, what=None, n=None, dt=None, t_range=None, moments=1, make_n_power_of_two=False
    ):
        """Create a timeseries from the timestamps provided"""
        t0, dt, data = binned_data(
            times,
            what=what,
            n=n,
            dv=dt,
            v_range=t_range,
            moments=moments,
            make_n_power_of_two=make_n_power_of_two,
        )
        return Timeseries(data, dt, t0)

    def times(self, endpoint=False):
        """Array of times associated with the datapoints

        Args:
            endpoint (bool): If true, returned array will have length data.size + 1
        """
        return self.t0 + self.dt * np.arange(self.data.size + int(endpoint))

    @property
    def duration(self):
        """The length of the timeseries in seconds"""
        return self.dt * self.data.size

    def crop(self, t_start=None, t_stop=None):
        """Crop data to time range

        Args:
            t_start (float | None): Time (in s) of first data to keep
            t_stop (float | None): Time (in s) of last data to keep
        Returns:
            Timeseries: The cropped timeseries
        """
        if self.size == 0:
            return Timeseries(np.empty(0), self.dt, self.t0)
        else:
            t_array = self.times()
            i_start, i_stop = 0, None
            if t_start is not None:
                i_start = int(np.argmin(np.abs(t_array - t_start)))
            if t_stop is not None:
                i_stop = int(np.argmin(np.abs(t_array - t_stop)))
            return Timeseries(self.data[i_start:i_stop], self.dt, t_array[i_start])

    def resample(self, dt):
        """Resample data to reduced time resolution

        Args:
            dt (float): The new sampling period. If not a multiple of the original sampling period,
                        it is rounded accordingly.
        Returns:
            Timeseries: The timeseries with new resolution
        """
        n = int(max(1, round(dt / self.dt)))
        size = n * int(self.data.size / n)
        data = np.reshape(self.data[:size], (-1, n)).mean(axis=1)
        return Timeseries(data, n * self.dt, self.t0)


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


class TimeBinPlot(_TimeBasePlot, ParticlePlotMixin, ParticleHistogramPlotMixin):
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
        timeseries=None,
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
            timeseries (Timeseries | dict[str, Timeseries]): Pre-binned timeseries data as alternative to timestamp-based particle data.
                                                             The dictionary must contain keys for each `kind` (e.g. `count`).
            time_range (tuple): Time range of particles to consider. If None, all particles are considered.
            time_offset (float): Time offset for x-axis is seconds, i.e. show values as `t-time_offset`.
            plot_kwargs (dict): Keyword arguments passed to the plot function, see :meth:`matplotlib.axes.Axes.plot`.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments

        """
        self.moment = moment
        kwargs = self._init_particle_mixin(**kwargs)
        kwargs = self._init_particle_histogram_mixin(**kwargs)

        super().__init__(time_range=time_range, time_offset=time_offset, on_y=kind, **kwargs)

        if bin_time is None and bin_count is None:
            bin_count = 100
        if bin_time is not None and bin_count is not None:
            raise ValueError("Only one of bin_time or bin_count may be specified.")
        self.bin_time = bin_time
        self.bin_count = bin_count
        self.relative = relative

        # Format plot axes
        self.axis(-1).set(xlabel=self.label_for(self.on_x), ylim=(0, None))
        if self.relative:
            for a in self.axes():
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
        if particles is not None or timeseries is not None:
            self.update(particles, mask=mask, timeseries=timeseries, autoscale=True)

    def _symbol_for(self, p):
        symbol = super()._symbol_for(p)
        if p != self.on_x and self.moment is not None and not self._count_based(p):
            # it is averaged
            symbol = f"$\\langle${symbol}$\\rangle$"
        return symbol

    def update(self, particles, mask=None, autoscale=False, *, timeseries=None):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (bool): Whether or not to perform autoscaling on all axes.
            timeseries (Timeseries | dict[str, Timeseries]): Pre-binned timeseries data as alternative to timestamp-based particle data.
                                                             The dictionary must contain keys for each `kind` (e.g. `count`).

        Returns:
            list: Changed artists
        """

        timeseries = self._ensure_particles_timeseries_data(particles, timeseries)

        # Particle timestamp based data
        ################################
        if particles is not None:

            # extract times
            times = self.prop(self.on_x).values(particles, mask, unit="s")

            # compute binned timeseries
            timeseries = {}
            for p in self.on_y_unique:
                count_based = self._count_based(p)
                if p in ("current", "charge"):
                    property = self.prop("q").values(particles, mask)
                elif count_based:
                    property = None
                else:
                    property = self.prop(p).values(particles, mask)
                timeseries[p] = Timeseries.from_timestamps(
                    times,
                    what=property,
                    n=self.bin_count,
                    dt=self.bin_time,
                    t_range=self.time_range,
                    moments=None if count_based else self.moment,
                )
                timeseries[p].nparticles = len(times)

        # Timeseries based data
        ########################
        else:
            # binned timeseries provided by user, apply time range
            for p in timeseries:
                nparticles = np.sum(timeseries[p].data)
                if self.time_range is not None:
                    timeseries[p] = timeseries[p].crop(*self.time_range)
                timeseries[p].nparticles = nparticles

        # update plots
        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                count_based = False
                edges = None
                for k, p in enumerate(pp):
                    count_based = self._count_based(p)
                    ts = timeseries[p]
                    values = ts.data.astype(np.float64)
                    edges = ts.times(endpoint=True)
                    dt = ts.dt

                    self.annotate(f"$\\Delta t_\\mathrm{{bin}} = {fmt(dt)}$")

                    if self.relative:
                        if not count_based:
                            raise ValueError(
                                f"Relative plots are only supported for kind 'count', 'rate', 'cumulative', 'charge' or 'current', but requested for {p}."
                            )
                        values /= ts.nparticles

                    if p in ("rate", "current"):
                        values /= dt

                    # post-processing expression wrappers
                    if wrap := self.on_y_expression[i][j][k]:
                        values = evaluate_expression_wrapper(wrap, p, values)

                    # display units
                    edges *= self.factor_for(self.on_x)
                    values *= self.factor_for(p)

                    # update plot
                    if p == "cumulative":
                        # steps open after last bin
                        values = np.concatenate(([0], np.cumsum(values)))
                    elif count_based:
                        # steps go back to zero after last bin
                        edges = np.append(edges, edges[-1])
                        values = np.concatenate(([0], values, [0]))
                    else:
                        edges = (edges[1:] + edges[:-1]) / 2
                    self.artists[i][j][k].set_data((edges, values))
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


class TimeFFTPlot(_TimeBasePlot, ParticlePlotMixin, ParticleHistogramPlotMixin):
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
        smoothing=None,
        mask=None,
        timeseries=None,
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
            log (bool | str): False, 'x', 'y' or True to plot none, the x-axis, y-axis or both on a log scale respectively.
            scaling (str | dict): Scaling of the FFT. Can be 'amplitude', 'power' or 'pdspp' or a dict with a scaling per property where
                                  `amplitude` (default for non-count based properties) scales the FFT magnitude to the amplitude,
                                  `power` (power density spectrum, default for count based properties) scales the FFT magnitude to power,
                                  `pdspp` (power density spectrum per particle) is simmilar to 'pds' but normalized to particle number.
            smoothing (int | None): If not None, uses Welch's method to compute a smoothened FFT with `2**smoothing` segments.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            timeseries (Timeseries | dict[str, Timeseries]): Pre-binned timeseries data as alternative to timestamp-based particle data.
                                                             The dictionary must contain keys for each `kind` (e.g. `count`).
            time_range (tuple): Time range of particles to consider. If None, all particles are considered.
            plot_kwargs (dict): Keyword arguments passed to the plot function, see :meth:`matplotlib.axes.Axes.plot`.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments

        """

        self._fmax = fmax
        self.relative = relative
        self._scaling = scaling
        if log is None:
            log = not relative
        self.smoothing = smoothing

        kwargs = self._init_particle_mixin(**kwargs)
        kwargs = self._init_particle_histogram_mixin(**kwargs)
        kwargs["_properties"] = defaults(
            kwargs.get("_properties"),
            f=Property("$f$", "Hz", description="Frequency"),
        )
        super().__init__(
            time_range=time_range, on_x=None, on_y=kind, **kwargs
        )  # handled manually

        # Format plot axes
        self.axis(-1).set(xlabel="$f/f_{rev}$" if self.relative else self.label_for("f"))
        for a in self.axflat:
            a.set(ylim=(0, None))
            a.set(**{f"{xy}scale": "log" for xy in "xy" if log in (True, xy)})

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
                autoscale=True,
            )

    def _get_scaling(self, key):
        if isinstance(self._scaling, str):
            return self._scaling.lower()
        if isinstance(self._scaling, dict) and key in self._scaling:
            return self._scaling[key].lower()
        return "power" if self._count_based(key) else "amplitude"

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

    def update(self, particles=None, mask=None, autoscale=False, *, timeseries=None):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (bool): Whether to perform autoscaling on all axes.
            timeseries (Timeseries | dict[str, Timeseries]): Pre-binned timeseries data as alternative to timestamp-based particle data.
                                                             The dictionary must contain keys for each `kind` (e.g. `count`).

        Returns:
            list: Changed artists
        """

        timeseries = self._ensure_particles_timeseries_data(particles, timeseries)

        # Particle timestamp based data
        ################################
        if particles is not None:

            # extract times
            times = self.prop("t").values(particles, mask, unit="s")
            fmax = self.fmax(particles)
            ppscale = len(times)

            # compute binned timeseries
            timeseries = {}
            for p in self.on_y_unique:
                prop = self.prop(p)
                property = None if self._count_based(p) else prop.values(particles, mask)
                timeseries[p] = Timeseries.from_timestamps(
                    times,
                    what=property,
                    dt=1 / (2 * fmax),
                    t_range=self.time_range,
                    make_n_power_of_two=True,  # to improve FFT performance, round up to next power of 2
                )

        # Timeseries based data
        ########################
        else:
            # binned timeseries provided by user, apply time range
            fmax = np.nan
            ppscale = 1 if "count" not in timeseries else np.sum(timeseries["count"].data)
            for p in timeseries:
                fmax = np.nanmax([fmax, self.fmax(default=timeseries[p].fs / 2)])
                if self.time_range is not None:
                    timeseries[p] = timeseries[p].crop(*self.time_range)

        # update plots
        changed = []
        ts = None
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):

                    # calculate FFT
                    ts = timeseries[p]
                    if self.smoothing is None:
                        freq = np.fft.rfftfreq(ts.size, d=ts.dt)
                        mag = np.abs(np.fft.rfft(ts.data, norm="forward"))
                        mag[
                            1:
                        ] *= 2  # one-sided spectrum contains only half the amplitude (except DC)
                    else:
                        # Welch's method for smoothing
                        freq, mag2 = scipy.signal.welch(
                            ts.data,
                            fs=ts.fs,
                            nperseg=ts.size // 2**self.smoothing,
                            scaling="spectrum",
                        )
                        mag = np.sqrt(2 * mag2)  # one-sided spectrum contains only half the power

                    # scale frequency according to user preferences
                    if self.relative:
                        freq *= 1 / self.frev(particles)
                    else:
                        freq *= self.factor_for("f")

                    # scale magnitude according to user preference
                    if p == "rate":
                        mag /= ts.dt
                    if self._get_scaling(p) == "amplitude":
                        # amplitude in units of p
                        mag *= self.factor_for(p)
                    elif self._get_scaling(p) in ("power", "pds", "pdspp"):
                        # power density spectrum in arb. unit
                        mag = mag**2
                        if self._get_scaling(p) == "pdspp":
                            mag /= ppscale  # per particle
                    else:
                        raise ValueError(f'Unknown scaling "{self._get_scaling(p)}"')

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
                    xlim = np.array((10.0 if log else 0.0, fmax))
                    if self.relative:
                        xlim /= self.frev(particles)
                    else:
                        xlim *= self.factor_for("f")
                    a.set_xlim(xlim)
                    if a.get_yscale() != "log":
                        a.set_ylim(0, None)

        self.annotate(f"$\\Delta t_\\mathrm{{bin}} = {fmt(ts.dt)}$" if ts is not None else "")

        return changed

    def _symbol_for(self, p):
        symbol = super()._symbol_for(p)
        if p not in "f":
            # it is the FFT of it
            symbol = symbol.strip("$")
            if self._get_scaling(p) == "amplitude":
                symbol = f"$\\hat{{{symbol}}}$"
            elif self._get_scaling(p) in ("power", "pds"):
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


class TimeIntervalPlot(_TimeBasePlot, ParticlePlotMixin, ParticleHistogramPlotMixin):
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
        super().__init__(time_range=time_range, on_x="dt", on_y=kind, **kwargs)

        if bin_time is None and bin_count is None:
            bin_count = 100
        self._bin_time = bin_time
        self._bin_count = bin_count
        self.relative = relative
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

        self.annotate(f"$\\Delta t_\\mathrm{{bin}} = {fmt(self.bin_time)}$")

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
                    counts = counts.astype(float)
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


class SpillQualityPlot(_TimeBasePlot, ParticlePlotMixin, MetricesMixin):
    """Plot variability of particle time on microscopic scale as function of time on macroscopic scale"""

    def __init__(
        self,
        particles=None,
        kind="cv",
        *,
        counting_dt=None,
        evaluate_dt=None,
        poisson=True,
        mask=None,
        timeseries=None,
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
            evaluate_dt (float): Time bin width for metric evaluation if evaluate_bins is None.
            poisson (bool): If true, indicate poisson limit.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            timeseries (Timeseries): Pre-binned timeseries data with particle counts
                                     as alternative to timestamp-based particle data.
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
            **self._metric_properties,
        )
        super().__init__(time_range=time_range, time_offset=time_offset, on_y=kind, **kwargs)

        self.counting_dt = counting_dt
        self.evaluate_dt = evaluate_dt

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
        if particles is not None or timeseries is not None:
            self.update(particles=particles, mask=mask, autoscale=True, timeseries=timeseries)

    def update(self, particles=None, mask=None, autoscale=False, *, timeseries=None):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (bool): Whether to perform autoscaling on all axes.
            timeseries (Timeseries): Pre-binned timeseries data with particle counts
                                     as alternative to timestamp-based particle data.

        Returns:
            Changed artists
        """

        timeseries = self._ensure_particles_timeseries_data(particles, timeseries)

        # Particle timestamp based data
        ################################
        if particles is not None:

            # extract times in range
            times = self._get_times_in_range(particles, mask)
            duration = np.max(times) - np.min(times)

            # bin into counting bins
            ncbins = int(np.ceil(duration / self.counting_dt)) if self.counting_dt else 10000
            timeseries = Timeseries.from_timestamps(times, n=ncbins, t_range=self.time_range)

        # Timeseries based data
        ########################
        else:

            # extract times in range
            if self.time_range is not None:
                timeseries = timeseries.crop(*self.time_range)

            # bin into counting bins
            if self.counting_dt:
                timeseries = timeseries.resample(self.counting_dt)

        # make 2D array by subdividing into evaluation bins
        counts, edges = timeseries.data, timeseries.times(endpoint=True)
        nebins = int(round(self.evaluate_dt / timeseries.dt)) if self.evaluate_dt else 100
        counts = counts[: int(len(counts) / nebins) * nebins].reshape((-1, nebins))
        edges = edges[: int(len(edges) / nebins + 1) * nebins : nebins]

        # annotate plot
        self.annotate(
            f"$\\Delta t_\\mathrm{{count}} = {fmt(timeseries.dt)}$\n"
            f"$\\Delta t_\\mathrm{{evaluate}} = {fmt(timeseries.dt * nebins)}$"
        )

        # display units
        edges = np.append(edges, edges[-1])
        edges *= self.factor_for(self.on_x)

        # update plots
        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):
                    # calculate metrics
                    F, F_poisson = self._calculate_metric(counts, p, axis=1)

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


class SpillQualityTimescalePlot(_TimeBasePlot, ParticlePlotMixin, MetricesMixin):
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
        timeseries=None,
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
            timeseries (Timeseries): Pre-binned timeseries data with particle counts
                                     as alternative to timestamp-based particle data.
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
        super().__init__(time_range=time_range, on_x="tbin", on_y=kind, **kwargs)

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
                    defaults_for("fill_between", std_kwargs, zorder=1.8, alpha=0.1, ls="-", lw=0)
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
        if particles is not None or timeseries is not None:
            self.update(
                particles=particles,
                mask=mask,
                autoscale=True,
                timeseries=timeseries,
                ignore_insufficient_statistics=ignore_insufficient_statistics,
            )

    def update(
        self,
        particles=None,
        mask=None,
        autoscale=False,
        *,
        timeseries=None,
        ignore_insufficient_statistics=False,
    ):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (bool): Whether to perform autoscaling on all axes.
            timeseries (Timeseries): Pre-binned timeseries data with particle counts
                                     as alternative to timestamp-based particle data.
            ignore_insufficient_statistics (bool): When set to True, the plot will include data with insufficient statistics.

        Returns:
            Changed artists
        """

        timeseries = self._ensure_particles_timeseries_data(particles, timeseries)

        counting_dt_min = self.counting_dt_min
        counting_dt_max = self.counting_dt_max

        def check_insufficient_statistics():
            if counting_dt_min > counting_dt_max or counting_dt_max > duration or ntotal < 1e4:
                print(
                    f"Warning: Data length ({duration:g} s), counting_dt_min ({counting_dt_min:g} s), "
                    f"counting_dt_max ({counting_dt_max:g} s) and/or count ({ntotal:g}) insufficient. "
                )
                if not ignore_insufficient_statistics:
                    print(f"Nothing plotted.")
                    return True

        # Particle timestamp based data
        ################################
        if particles is not None:

            # extract times in range
            times = self._get_times_in_range(particles, mask)
            ntotal = times.size
            duration = np.max(times) - np.min(times)

            # determine timescales
            if counting_dt_min is None:
                counting_dt_min = (
                    50 * duration / ntotal
                )  # at least 50 particles per bin (on average)
            if counting_dt_max is None:
                counting_dt_max = duration / 50  # at least 50 bins to calculate metric
            if check_insufficient_statistics():
                return

            # determine bins
            bins_min = int(duration / counting_dt_max + 1)
            bins_max = int(duration / counting_dt_min + 1)
            if self.log:
                ncbins = 1 / np.geomspace(1 / bins_min, 1 / bins_max, 100)
            else:
                ncbins = 1 / np.linspace(1 / bins_min, 1 / bins_max, 100)
            ncbins = np.unique(ncbins.astype(int))

            # bin data
            TS = [Timeseries.from_timestamps(times, n=n) for n in ncbins]

        # Timeseries based data
        ########################
        else:

            # extract times in range
            if self.time_range is not None:
                timeseries = timeseries.crop(*self.time_range)
            duration = timeseries.duration
            ntotal = np.sum(timeseries.data)

            # determine timescales
            if counting_dt_min is None or counting_dt_min < 1 / timeseries.fs:
                counting_dt_min = timeseries.dt  # at least resolution of data
            if counting_dt_max is None:
                counting_dt_max = duration / 50  # at least 50 bins to calculate metric
            if check_insufficient_statistics():
                return

            # determine bins
            rebin_min = int(round(counting_dt_min * timeseries.fs))
            rebin_max = int(round(counting_dt_max * timeseries.fs))
            if self.log:
                rebins = np.geomspace(rebin_min, rebin_max, 100)
            else:
                rebins = np.linspace(rebin_min, rebin_max, 100)
            rebins = np.unique(rebins.astype(int))

            # re-bin data
            TS = [timeseries.resample(r * timeseries.dt) for r in rebins]

        # Metric calculation
        #####################
        F = {m: np.empty(len(TS)) for m in self.on_y_unique}
        F_std = {m: np.empty(len(TS)) for m in self.on_y_unique}
        F_poisson = {m: np.empty(len(TS)) for m in self.on_y_unique}

        for i, ts in enumerate(TS):
            # reshape into evaluation bins using a sliding window
            stride = min(self.counting_bins_per_evaluation or ts.size, ts.size)
            NN = np.lib.stride_tricks.sliding_window_view(ts.data, stride)
            if NN.shape[0] > 10000:  # not more than this many windows
                NN = NN[:: NN.shape[0] // 10000, :]

            for metric in F.keys():
                # calculate metrics
                v, lim = self._calculate_metric(NN, metric, axis=1)
                F[metric][i] = np.nanmean(v)
                F_std[metric][i] = np.nanstd(v) or np.nan
                F_poisson[metric][i] = np.nanmean(lim)

        DT = self.factor_for(self.on_x) * np.array([ts.dt for ts in TS])

        # annotate plot
        # f'$\\langle\\dot{{N}}\\rangle = {fmt(ntotal/duration, "1/s")}$\n'
        self.annotate(
            "$\\Delta t_\\mathrm{evaluate} = "
            + (
                f"{self.counting_bins_per_evaluation:g}\\,\\Delta t_\\mathrm{{count}}$"
                if self.counting_bins_per_evaluation
                else f"{fmt(duration)}$"
            )
        )

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


__all__ = PUBLIC_SECTION_END()
