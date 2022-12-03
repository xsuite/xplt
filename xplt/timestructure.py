#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting particle arrival times

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-24"


import matplotlib as mpl
import numpy as np

from .base import defaults, XParticlePlot

c0 = 299792458  # speed of light in m/s


def binned_timeseries(times, n, what=None):
    """Get binned timeseries with equally spaced time bins

    From the particle arrival times (non-equally distributed timestamps), a timeseries with equally
    spaced time bins is derived. The time bin size is determined based on the number of bins.
    The parameter `what` determines what is returned for the timeseries. By default (what=None), the
    number of particles arriving within each time bin is returned. Alternatively, a particle property
    can be passed as array, in which case that property is averaged over all particles arriving within
    the respective bin (or 0 if no particles arrive within a time bin).

    Args:
        times: Array of particle arrival times.
        what: Array of associated data or None. Must have same shape as times. See above.
        n: Number of bins.
        range: Tuple of (min, max) time values to consider. If None, the range is determined from the data.

    Returns:
        The timeseries as tuple (t_min, dt, values) where
        t_min is the start time of the timeseries data,
        dt is the time bin width and
        values are the values of the timeseries as array of length n.
    """

    # Note: The code below was optimized to run much faster than an ordinary
    # np.histogram, which quickly slows down for large datasets.
    # If you intend to change something here, make sure to benchmark it!

    t_min = np.min(times)
    dt = (np.max(times) - t_min) / n
    # count timestamps in bins
    bins = ((times - t_min) / dt).astype(int)
    # bins are i*dt <= t < (i+1)*dt where i = 0 .. n-1
    bins = np.clip(bins, None, n - 1)  # but for the last bin use t <= n*dt
    # count particles per time bin
    counts = np.bincount(bins)  # , minlength=n)[:n]

    if what is None:
        # Return particle counts
        return t_min, dt, counts

    else:
        # Return 'what' averaged
        v = np.zeros(n)
        # sum up 'what' for all the particles in each bin
        np.add.at(v, bins, what)
        # divide by particle count to get mean (default to 0)
        v[counts > 0] /= counts[counts > 0]
        return t_min, dt, v


class TimePlot(XParticlePlot):
    def __init__(
        self,
        particles=None,
        kind="x+y",
        *,
        mask=None,
        plot_kwargs=None,
        grid=True,
        ax=None,
        data_units=None,
        display_units=None,
        twiss=None,
        beta=None,
        frev=None,
        circumference=None,
        wrap_zeta=False,
        **subplots_kwargs,
    ):
        """
        A plot of particle properties as function of times.

        The plot is based on the particle arrival time, which is:
            - For circular lines: at_turn / frev - zeta / beta / c0
            - For linear lines: zeta / beta / c0

        Args:
            particles: Particles data to plot.
            kind: Defines the properties to plot.
                 This can be a nested list or a separated string or a mixture of lists and strings where
                 the first list level (or separator ``,``) determines the subplots,
                 the second list level (or separator ``-``) determines any twinx-axes,
                 and the third list level (or separator ``+``) determines plots on the same axis.

            mask: An index mask to select particles to plot. If None, all particles are plotted.
            plot_kwargs: Keyword arguments passed to the plot function.
            grid: If True, show grid lines.
            ax: Axes to plot on. If None, a new figure is created.
            data_units (dict, optional): Units of the data. If None, the units are determined from the data.
            display_units (dict, optional): Units to display the data in. If None, the units are determined from the data.
            twiss (dict, optional): Twiss parameters (alfx, alfy, betx and bety) to use for conversion to normalized phase space coordinates.
            beta (float, optional): Relativistic beta of particles. Defaults to particles.beta0.
            frev (float, optional): Revolution frequency of circular line for calculation of particle time.
            circumference (float, optional): Path length of circular line if frev is not given.
            wrap_zeta: If set, wrap the zeta-coordinate plotted at the machine circumference. Either pass the circumference directly or set this to True to use the circumference from twiss.
            subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.

        """
        super().__init__(
            data_units=data_units,
            display_units=display_units,
            twiss=twiss,
            beta=beta,
            frev=frev,
            circumference=circumference,
            wrap_zeta=wrap_zeta,
        )

        # parse kind string
        self.kind = self._parse_nested_list_string(kind)

        # initialize figure with n subplots
        nntwins = [len(tw) - 1 for tw in self.kind]
        self._init_axes(ax, len(self.kind), 1, nntwins, grid, sharex="col", **subplots_kwargs)

        # Format plot axes
        self.axis_for(-1).set(xlabel=self.label_for("t"))

        # create plot elements
        def create_artists(i, j, k, a, p):
            kwargs = defaults(plot_kwargs, marker=".", ls="", label=self.label_for(p, unit=False))
            return a.plot([], [], **kwargs)[0]

        self._init_artists(self.kind, create_artists)

        # set data
        if particles is not None:
            self.update(particles, mask=mask, autoscale=True)

    def update(self, particles, mask=None, autoscale=False):
        """Update plot with new data

        Args:
            particles: Particles data to plot.
            mask: An index mask to select particles to plot. If None, all particles are plotted.
            autoscale: Whether or not to perform autoscaling on all axes.
        """

        times = self._get_masked(particles, "t", mask)
        order = np.argsort(times)
        times = times[order] * self.factor_for("t")

        changed = []
        for i, ppp in enumerate(self.kind):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):
                    values = self._get_masked(particles, p, mask)
                    values = values[order] * self.factor_for(p)
                    self.artists[i][j][k].set_data((times, values))
                    changed.append(self.artists[i][j][k])

                if autoscale:
                    a = self.axis_for(i, j)
                    a.relim()
                    a.autoscale()

        return changed


class TimeBinPlot(XParticlePlot):
    def __init__(
        self,
        particles=None,
        kind="count",
        *,
        bin_time=None,
        bin_count=None,
        relative=False,
        mask=None,
        plot_kwargs=None,
        grid=True,
        ax=None,
        data_units=None,
        display_units=None,
        twiss=None,
        beta=None,
        frev=None,
        circumference=None,
        wrap_zeta=False,
        **subplots_kwargs,
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
                particles: Particles data to plot.
                kind (str, optional): What to plot as function of time. Can be 'count' (default),
                    'rate', 'cumulative', or a particle property to average.
                bin_time: Time bin width if bin_count is None.
                bin_count: Number of bins if bin_time is None.
                relative: If True, plot relative numbers normalized to total count.
                    If what is a particle property, this has no effect.
                mask: An index mask to select particles to plot. If None, all particles are plotted.
                plot_kwargs: Keyword arguments passed to the plot function.
                grid: If True, show grid lines.
                ax: Axes to plot on. If None, a new figure is created.
                data_units (dict, optional): Units of the data. If None, the units are determined from the data.
                display_units (dict, optional): Units to display the data in. If None, the units are determined from the data.
                twiss (dict, optional): Twiss parameters (alfx, alfy, betx and bety) to use for conversion to normalized phase space coordinates.
                beta (float, optional): Relativistic beta of particles. Defaults to particles.beta0.
                frev (float, optional): Revolution frequency of circular line for calculation of particle time.
                circumference (float, optional): Path length of circular line if frev is not given.
                wrap_zeta: If set, wrap the zeta-coordinate plotted at the machine circumference. Either pass the circumference directly or set this to True to use the circumference from twiss.
                subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.

        """
        super().__init__(
            data_units=data_units,
            display_units=display_units,
            twiss=twiss,
            beta=beta,
            frev=frev,
            circumference=circumference,
            wrap_zeta=wrap_zeta,
        )

        if bin_time is None and bin_count is None:
            bin_count = 100
        self.kind = self._parse_nested_list_string(kind)
        self.bin_time = bin_time
        self.bin_count = bin_count
        self.relative = relative

        # initialize figure with n subplots
        nntwins = [len(tw) - 1 for tw in self.kind]
        self._init_axes(ax, len(self.kind), 1, nntwins, grid, sharex="col", **subplots_kwargs)

        # Format plot axes
        self.axis_for(-1).set(xlabel=self.label_for("t"), ylim=(0, None))
        if self.relative:
            for a in self.axflat:
                a.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

        # Create plot elements
        def create_artists(i, j, k, ax, p):
            kwargs = defaults(plot_kwargs, lw=1, label=self.label_for(p, unit=False))
            if p in ("count", "rate", "cumulative"):
                kwargs = defaults(kwargs, drawstyle="steps-pre")
            return ax.plot([], [], **kwargs)[0]

        self._init_artists(self.kind, create_artists)

        # set data
        if particles is not None:
            self.update(particles, mask=mask, autoscale=True)

    def update(self, particles, mask=None, autoscale=False):
        """Update plot with new data

        Args:
            particles: Particles data to plot.
            mask: An index mask to select particles to plot. If None, all particles are plotted.
            autoscale: Whether or not to perform autoscaling on all axes.
        """

        # extract times
        times = self._get_masked(particles, "t", mask)

        # re-sample times into equally binned time series
        n = self.bin_count or int(np.ceil((np.max(times) - np.min(times)) / self.bin_time))

        # update plots
        changed = []
        for i, ppp in enumerate(self.kind):
            for j, pp in enumerate(ppp):
                count_based = False
                for k, p in enumerate(pp):
                    count_based = p in ("count", "rate", "cumulative")
                    if count_based:
                        property = None
                    else:
                        property = self._get_masked(particles, p, mask)

                    t_min, dt, timeseries = binned_timeseries(times, n, property)
                    timeseries = timeseries.astype(np.float)
                    edges = np.linspace(t_min, t_min + dt * n, n + 1)

                    if self.relative:
                        if not count_based:
                            raise ValueError(
                                "Relative plots are only supported for kind 'count', 'rate' or 'cumulative'."
                            )
                        timeseries /= len(times)

                    if p == "rate":
                        timeseries /= dt

                    # target units
                    edges *= self.factor_for("t")
                    if not count_based:
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
                    a = self.axis_for(i, j)
                    a.relim()
                    a.autoscale()
                    if count_based:
                        a.set(ylim=(0, None))

        return changed

    def data_unit_for(self, p):
        if p in ("count", "cumulative"):
            return "1"
        if p == "rate":
            return "1/s"
        return super().data_unit_for(p)

    def label_for(self, *pp, unit=True):
        def texify(label):
            if label == "count":
                return "\\mathrm{Particles}"
            if label == "rate":
                return "\\mathrm{Particle~rate}"
            if label == "cumulative":
                return "\\mathrm{Particles~(cumulative)}"

        return super().label_for(*pp, unit=unit, texify=texify)


class TimeFFTPlot(XParticlePlot):
    def __init__(
        self,
        particles=None,
        kind="count",
        *,
        fmax=None,
        relative=False,
        log=True,
        scaling="amplitude",
        mask=None,
        plot_kwargs=None,
        grid=True,
        ax=None,
        data_units=None,
        display_units=None,
        twiss=None,
        beta=None,
        frev=None,
        circumference=None,
        wrap_zeta=False,
        **subplots_kwargs,
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
                particles: Particles data to plot.
                kind (str, optional): What to make the FFT over. Can be 'count' (default), or a particle property (in which case averaging applies).
                fmax (float): Maximum frequency (in Hz) to plot.
                relative (bool): If True, plot relative frequencies (f/frev) instead of absolute frequencies (f).
                log: If True, plot on a log scale.
                scaling: Scaling of the FFT. Can be 'amplitude' (default) or 'pds'.
                mask: An index mask to select particles to plot. If None, all particles are plotted.
                plot_kwargs: Keyword arguments passed to the plot function.
                grid: If True, show grid lines.
                ax: Axes to plot on. If None, a new figure is created.
                data_units (dict, optional): Units of the data. If None, the units are determined from the data.
                display_units (dict, optional): Units to display the data in. If None, the units are determined from the data.
                twiss (dict, optional): Twiss parameters (alfx, alfy, betx and bety) to use for conversion to normalized phase space coordinates.
                beta (float, optional): Relativistic beta of particles. Defaults to particles.beta0.
                frev (float, optional): Revolution frequency of circular line for calculation of particle time.
                circumference (float, optional): Path length of circular line if frev is not given.
                wrap_zeta: If set, wrap the zeta-coordinate plotted at the machine circumference. Either pass the circumference directly or set this to True to use the circumference from twiss.
                subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.

        """
        if fmax is None:
            raise ValueError("fmax must be specified.")

        display_units = defaults(display_units, f="Hz" if log else "kHz")
        super().__init__(
            data_units=data_units,
            display_units=display_units,
            twiss=twiss,
            beta=beta,
            frev=frev,
            circumference=circumference,
            wrap_zeta=wrap_zeta,
        )

        self.kind = self._parse_nested_list_string(kind)
        self.fmax = fmax
        self.relative = relative
        self.scaling = scaling

        # initialize figure with n subplots
        nntwins = [len(tw) - 1 for tw in self.kind]
        self._init_axes(ax, len(self.kind), 1, nntwins, grid, sharex="col", **subplots_kwargs)

        # Format plot axes
        self.axis_for(-1).set(
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
            kwargs = defaults(plot_kwargs, lw=1, label=self.label_for(p, unit=False))
            return ax.plot([], [], **kwargs)[0]

        self._init_artists(self.kind, create_artists)

        # set data
        if particles is not None:
            self.update(particles, mask=mask, autoscale=True)

    def update(self, particles, mask=None, autoscale=False):
        """Update plot with new data

        Args:
            particles: Particles data to plot.
            mask: An index mask to select particles to plot. If None, all particles are plotted.
            autoscale: Whether or not to perform autoscaling on all axes.
        """

        # extract times and associated property
        times = self._get_masked(particles, "t", mask)

        # re-sample times into equally binned time series
        dt = 1 / 2 / self.fmax
        n = int(np.ceil((np.max(times) - np.min(times)) / dt))
        # to improve FFT performance, round up to next power of 2
        self.nbins = n = 1 << (n - 1).bit_length()
        freq = np.fft.rfftfreq(n, d=dt)[1:]
        if self.relative:
            freq /= self.frev(particles)
        else:
            freq *= self.factor_for("f")

        # update plots
        changed = []
        for i, ppp in enumerate(self.kind):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):
                    count_based = p == "count"
                    if count_based:
                        property = None
                    else:
                        property = self._get_masked(particles, p, mask)

                    # compute binned timeseries
                    t_min, dt, timeseries = binned_timeseries(times, n, property)

                    # calculate fft without DC component
                    mag = np.abs(np.fft.rfft(timeseries))[1:]
                    if self.scaling.lower() == "amplitude":
                        # amplitude in units of particle counts
                        mag *= 2 / len(timeseries)
                    elif self.scaling.lower() == "pds":
                        # power density spectrum in a.u.
                        mag = mag**2

                    # update plot
                    self.artists[i][j][k].set_data(freq, mag)
                    changed.append(self.artists[i][j][k])

                if autoscale:
                    a = self.axis_for(i, j)
                    a.relim()
                    a.autoscale()
                    log = a.get_xscale() == "log"
                    xlim = np.array((10.0, self.fmax) if log else (0.0, self.fmax))
                    if self.relative:
                        xlim /= self.frev(particles)
                    else:
                        xlim *= self.factor_for("f")
                    a.set_xlim(xlim)
                    if a.get_yscale() != "log":
                        a.set_ylim(0, None)

        return changed

    def data_unit_for(self, p):
        if p == "count":
            return "1"
        return super().data_unit_for(p)

    def display_unit_for(self, p):
        if p != "f" and self.scaling.lower() == "pds":
            return "a.u."
        return super().display_unit_for(p)

    def label_for(self, *pp, unit=True):
        def texify(label):
            if label == "f":
                return  # don't change the x-axis label
            if self.scaling.lower() == "amplitude":
                if label == "count":
                    return "\\mathrm{Particles}"
                else:
                    return "\\hat{" + label + "}"
            elif self.scaling.lower() == "pds":
                return "|\\mathrm{FFT}|^2"

        return super().label_for(*pp, unit=unit, texify=texify)

    def plot_harmonics(self, v, dv=0, *, n=20, inverse=False, **plot_kwargs):
        """Add vertical lines or spans indicating the location of values or spans and their harmonics

        Args:
            v (float or list of float): Value or list of values.
            dv (float or list of float, optional): Width or list of widths centered around value(s).
            n (int): Number of harmonics to plot.
            inverse (bool): If true, plot harmonics of n/(v±dv) instead of n*(v±dv). Useful to plot frequency harmonics in time domain and vice-versa.
            plot_kwargs: Keyword arguments to be passed to plotting method
        """
        for a in self.axflat:
            super().plot_harmonics(a, v, dv, n=n, inverse=inverse, **plot_kwargs)


class TimeIntervalPlot(XParticlePlot):
    def __init__(
        self,
        particles=None,
        *,
        tmax=None,
        bin_time=None,
        bin_count=None,
        log=True,
        mask=None,
        plot_kwargs=None,
        grid=True,
        ax=None,
        data_units=None,
        display_units=None,
        beta=None,
        frev=None,
        circumference=None,
        **subplots_kwargs,
    ):
        """
        A histogram plot of particle arrival intervals (i.e. delay between consecutive particles).

        The plot is based on the particle arrival time, which is:
            - For circular lines: at_turn / frev - zeta / beta / c0
            - For linear lines: zeta / beta / c0

        Useful to plot time structures of particles loss, such as spill structures.

        Args:
                particles: Particles data to plot.
                tmax: Maximum interval (in s) to plot.
                bin_time: Time bin width if bin_count is None.
                bin_count: Number of bins if bin_time is None.
                log: If True, plot on a log scale.
                mask: An index mask to select particles to plot. If None, all particles are plotted.
                plot_kwargs: Keyword arguments passed to the plot function.
                grid: If True, show grid lines.
                ax: Axes to plot on. If None, a new figure is created.
                data_units (dict, optional): Units of the data. If None, the units are determined from the data.
                display_units (dict, optional): Units to display the data in. If None, the units are determined from the data.
                beta (float, optional): Relativistic beta of particles. Defaults to particles.beta0.
                frev (float, optional): Revolution frequency of circular line for calculation of particle time.
                circumference (float, optional): Path length of circular line if frev is not given.
                subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.

        """
        if tmax is None:
            raise ValueError("tmax must be specified.")

        super().__init__(
            data_units=data_units,
            display_units=display_units,
            beta=beta,
            frev=frev,
            circumference=circumference,
        )

        if bin_time is None and bin_count is None:
            bin_count = 100
        self._bin_time = bin_time
        self._bin_count = bin_count
        self.tmax = tmax

        # initialize figure with 1 subplot
        self._init_axes(ax, 1, 1, [0], grid, **subplots_kwargs)

        # Format plot axes
        ax = self.axis_for(-1)
        ax.set(
            xlabel="Delay between consecutive particles " + self.label_for("t"),
            xlim=(self.bin_time if log else 0, self.tmax * self.factor_for("t")),
            ylabel=f"Occurrences",
        )
        if log:
            ax.set(xscale="log", yscale="log")
        else:
            ax.set(ylim=(0, None))

        # Create plot elements
        kwargs = defaults(plot_kwargs, lw=1)
        (self.artist,) = self.ax.step([], [], **kwargs)

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
            particles: Particles data to plot.
            mask: An index mask to select particles to plot. If None, all particles are plotted.
            autoscale: Whether or not to perform autoscaling on all axes.
        """

        # extract times
        times = self._get_masked(particles, "t", mask)
        delay = self.factor_for("t") * np.diff(sorted(times))

        # calculate and plot histogram
        counts, edges = np.histogram(
            delay, bins=self.bin_count, range=(0, self.bin_count * self.bin_time)
        )
        steps = (np.append(edges, edges[-1]), np.concatenate(([0], counts, [0])))
        self.artist.set_data(steps)

        if autoscale:
            ax = self.axis_for(-1)
            ax.relim()
            ax.autoscale()
            if not ax.get_yscale() == "log":
                ax.set(ylim=(0, None))

    def plot_harmonics(self, v, dv=0, *, n=20, inverse=False, **plot_kwargs):
        """Add vertical lines or spans indicating the location of values or spans and their harmonics

        Args:
            v (float or list of float): Value or list of values.
            dv (float or list of float, optional): Width or list of widths centered around value(s).
            n (int): Number of harmonics to plot.
            inverse (bool): If true, plot harmonics of n/(v±dv) instead of n*(v±dv). Useful to plot frequency harmonics in time domain and vice-versa.
            plot_kwargs: Keyword arguments to be passed to plotting method
        """
        return super().plot_harmonics(
            self.axis_for(-1), v, dv, n=n, inverse=inverse, **plot_kwargs
        )


class TimeVariationPlot(XParticlePlot):
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
        grid=True,
        ax=None,
        data_units=None,
        display_units=None,
        beta=None,
        frev=None,
        circumference=None,
        **subplots_kwargs,
    ):
        """
        Plot for variability of particles arriving as function of arrival time

        The plot is based on the particle arrival time, which is:
            - For circular lines: at_turn / frev - zeta / beta / c0
            - For linear lines: zeta / beta / c0

        Useful to plot time structures of particles loss, such as spill structures.

        The following metrics are implemented:
            cv: Coefficient of variation
                cv = std(N)/mean(N)
            duty: Spill duty factor
                F = mean(N)**2 / mean(N**2)

        Args:
            particles: Particles data to plot.
            metric (str): Metric to plot. See above for list of implemented metrics.
            counting_dt: Time bin width for counting if counting_bins is None.
            counting_bins: Number of bins if counting_dt is None.
            evaluate_dt: Time bin width for metric evaluation if evaluate_bins is None.
            evaluate_bins: Number of bins if evaluate_dt is None.
            poisson (bool): If true, indicate poisson limit.

            mask: An index mask to select particles to plot. If None, all particles are plotted.
            plot_kwargs: Keyword arguments passed to the plot function.
            grid: If True, show grid lines.
            ax: Axes to plot on. If None, a new figure is created.
            data_units (dict, optional): Units of the data. If None, the units are determined from the data.
            display_units (dict, optional): Units to display the data in. If None, the units are determined from the data.
            beta (float, optional): Relativistic beta of particles. Defaults to particles.beta0.
            frev (float, optional): Revolution frequency of circular line for calculation of particle time.
            circumference (float, optional): Path length of circular line if frev is not given.
            subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.

        """
        super().__init__(
            data_units=data_units,
            display_units=display_units,
            beta=beta,
            frev=frev,
            circumference=circumference,
        )

        if counting_dt is None and counting_bins is None:
            counting_bins = 100 * 100
        if evaluate_dt is None and evaluate_bins is None:
            evaluate_bins = 100
        self.kind = self._parse_nested_list_string(metric)
        self.counting_dt = counting_dt
        self.counting_bins = counting_bins
        self.evaluate_dt = evaluate_dt
        self.evaluate_bins = evaluate_bins

        # initialize figure with n subplots
        nntwins = [len(tw) - 1 for tw in self.kind]
        self._init_axes(ax, len(self.kind), 1, nntwins, grid, sharex="col", **subplots_kwargs)

        # Format plot axes
        self.axis_for(-1).set(xlabel=self.label_for("t"), ylim=(0, None))
        for i, ppp in enumerate(self.kind):
            for j, pp in enumerate(ppp):
                a = self.axis_for(i, j)
                if np.all(np.array(pp) == "duty"):
                    a.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

        # Create plot elements
        def create_artists(i, j, k, ax, p):
            kwargs = defaults(plot_kwargs, lw=1, label=self.label_for(p, unit=False))
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

        self._init_artists(self.kind, create_artists)

        # set data
        if particles is not None:
            self.update(particles, mask=mask, autoscale=True)

    def update(self, particles, mask=None, autoscale=False):
        """Update plot with new data

        Args:
            particles: Particles data to plot.
            mask: An index mask to select particles to plot. If None, all particles are plotted.
            autoscale: Whether or not to perform autoscaling on all axes.
        """

        # extract times
        times = self._get_masked(particles, "t", mask)

        # re-sample times into equally binned time series
        bin_time = self.counting_dt or (np.max(times) - np.min(times)) / self.counting_bins
        ncbins = self.counting_bins or int(
            np.ceil((np.max(times) - np.min(times)) / self.counting_dt)
        )
        if self.evaluate_bins is not None:
            nebins = int(ncbins / self.evaluate_bins)
        else:
            nebins = int(self.evaluate_dt / bin_time)

        # update plots
        changed = []
        for i, ppp in enumerate(self.kind):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):
                    # bin into counting bins
                    t_min, dt, counts = binned_timeseries(times, ncbins)
                    edges = np.linspace(t_min, t_min + dt * ncbins, ncbins + 1)

                    # make 2D array by subdividing into evaluation bins
                    N = counts = counts[: int(len(counts) / nebins) * nebins].reshape(
                        (-1, nebins)
                    )
                    edges = edges[: int(len(edges) / nebins + 1) * nebins : nebins]

                    # calculate metrics
                    if p == "cv":
                        F = np.std(N, axis=1) / np.mean(N, axis=1)
                        F_poisson = 1 / np.mean(N, axis=1) ** 0.5
                    elif p == "duty":
                        F = np.mean(N, axis=1) ** 2 / np.mean(N**2, axis=1)
                        F_poisson = 1 / (1 + 1 / np.mean(N, axis=1))
                    else:
                        raise ValueError(f"Unknown metric {p}")

                    # update plot
                    step, pstep = self.artists[i][j][k]
                    edges = self.factor_for("t") * np.append(edges, edges[-1])
                    steps = np.concatenate(([0], F, [0]))
                    step.set_data((edges, steps))
                    changed.append(step)
                    if pstep:
                        steps = np.concatenate(([0], F_poisson, [0]))
                        pstep.set_data((edges, steps))
                        changed.append(pstep)

                if autoscale:
                    a = self.axis_for(i, j)
                    a.relim()
                    a.autoscale()
                    a.set(ylim=(0, None))
        return changed

    def data_unit_for(self, p):
        if p in ("cv", "duty"):
            return "1"
        return super().data_unit_for(p)

    def label_for(self, *pp, unit=True):
        def texify(label):
            if label == "cv":
                return "\\mathrm{Coefficient~of~variation}~ c_v=\\sigma/\\mu"
            elif label == "duty":
                return (
                    "\\mathrm{Spill~duty~factor}~ F=\\langle N \\rangle^2/\\langle N^2 \\rangle"
                )

        return super().label_for(*pp, unit=unit, texify=texify)
