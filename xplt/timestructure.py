#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting particle arrival times

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-24"


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

import pint

from .base import Xplot, style, get

c0 = 299792458  # speed of light in m/s


class _TimestructurePlotMixin:
    def __init__(self, beta, frev=None, *args, **kwargs):
        """A mixing for plots which are based on the particle arrival time

        Args:
            beta: Relativistic beta of particles.
            frev: Revolution frequency of circular line. If None for linear lines.
        """
        if beta is None:
            raise ValueError("beta is a required parameter.")
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.frev = frev

    def time(self, particles, mask=None):
        """Get particle arrival times

        Args:
            particles: Particles data to plot.
            mask: An index mask to select particles to plot. If None, all particles are plotted.

        """
        turn = get(particles, "at_turn")
        zeta = get(particles, "zeta")
        if mask is not None:
            turn = turn[mask]
            zeta = zeta[mask]

        time = -zeta / self.beta / c0  # zeta>0 means early; zeta<0 means late
        if self.frev is not None:
            time = time + turn / self.frev
        elif np.any(turn > 0):
            raise ValueError("frev is required for non-circular lines where turn > 0.")

        return np.array(sorted(time))


class TimeHistPlot(_TimestructurePlotMixin, Xplot):
    def __init__(
        self,
        particles=None,
        *,
        beta=None,
        frev=None,
        bin_time=None,
        bin_count=None,
        plot="counts",
        ax=None,
        mask=None,
        relative=False,
        range=None,
        display_units=None,
        step_kwargs=None,
        grid=True,
        **subplots_kwargs,
    ):
        """
        A histogram plot of particle arrival times.

        The plot is based on the particle arrival time, which is:
            - For circular lines: at_turn / frev + zeta / beta / c0
            - For linear lines: zeta / beta / c0

        Useful to plot time structures of particles loss, such as spill structures.

        Args:
                particles: Particles data to plot.
                beta: Relativistic beta of particles.
                frev: Revolution frequency of circular line. If None for linear lines.
                bin_time: Time bin width if bin_count is None.
                bin_count: Number of bins if bin_time is None.
                plot: Plot type. Can be 'counts' (default), 'rate' or 'cumulative'.
                ax: An axes to plot onto. If None, a new figure is created.
                mask: An index mask to select particles to plot. If None, all particles are plotted.
                relative: If True, plot relative numbers normalized to total count.
                range: A tuple of (min, max) time values defining the histogram range.
                display_units: Dictionary with units for parameters.
                step_kwargs: Keyword arguments passed to matplotlib.pyplot.step() plot.
                grid (bool): En- or disable showing the grid
                subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.

        """
        super().__init__(beta, frev, display_units=display_units)

        if bin_time is None and bin_count is None:
            bin_count = 100
        self.plot = plot
        self.bin_time = bin_time
        self.bin_count = bin_count
        self.relative = relative
        self.range = range

        # Create plot axes
        if ax is None:
            _, ax = plt.subplots(**subplots_kwargs)
        self.ax = ax
        self.fig = self.ax.figure

        # Create distribution plots
        kwargs = style(step_kwargs, lw=1)
        (self.artist_hist,) = self.ax.step([], [], **kwargs)
        self.ax.set(xlabel=self.label_for("t"), ylim=(0, None))
        self.ax.grid(grid)
        if self.relative:
            self.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

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
        times = self.factor_for("t") * self.time(particles, mask=mask)

        # histogram settings
        bin_time = self.bin_time or (times[-1] - times[0]) / self.bin_count
        range = self.range or (np.min(times), np.max(times))
        nbins = int(np.ceil((range[1] - range[0]) / bin_time))
        range = (range[0], range[0] + nbins * bin_time)  # ensure exact bin width

        # calculate and plot histogram
        weights = np.ones_like(times)
        if self.plot == "rate":
            weights /= bin_time
        if self.relative:
            weights /= len(times)
        counts, edges = np.histogram(times, bins=nbins, weights=weights, range=range)

        # update plot
        if self.plot == "cumulative":
            counts = np.cumsum(counts)
            steps = (edges, np.concatenate(([0], counts)))
        else:
            steps = (np.append(edges, edges[-1]), np.concatenate(([0], counts, [0])))
        self.artist_hist.set_data(steps)

        if autoscale:
            self.ax.relim()
            self.ax.autoscale()
            self.ax.set(ylim=(0, None))

        # label
        ylabel = "Particle" + (" fraction" if self.relative else "s")
        if self.plot == "rate":
            ylabel += " / s"
        elif self.plot != "cumulative":
            ylabel += (
                f"\nper ${pint.Quantity(bin_time, 's').to_compact():~gL}$ interval"
            )
        self.ax.set(ylabel=ylabel)


class TimeFFTPlot(_TimestructurePlotMixin, Xplot):
    def __init__(
        self,
        particles=None,
        *,
        beta=None,
        frev=None,
        fmax=None,
        log=True,
        ax=None,
        mask=None,
        display_units=None,
        plot_kwargs=None,
        scaling="amplitude",
        **subplots_kwargs,
    ):
        """
        A frequency plot of particle arrival times.

        The plot is based on the particle arrival time, which is:
            - For circular lines: at_turn / frev + zeta / beta / c0
            - For linear lines: zeta / beta / c0

        Useful to plot time structures of particles loss, such as spill structures.

        Args:
                particles: Particles data to plot.
                beta: Relativistic beta of particles.
                frev: Revolution frequency of circular line. If None for linear lines.
                fmax: Maximum frequency (in Hz) to plot.
                log: If True, plot on a log scale.
                ax: An axes to plot onto. If None, a new figure is created.
                mask: An index mask to select particles to plot. If None, all particles are plotted.
                display_units: Dictionary with units for parameters.
                plot_kwargs: Keyword arguments passed to matplotlib.pyplot.plot() plot.
                scaling: Scaling of the FFT. Can be 'amplitude' (default) or 'pds'.
                subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.

        """
        super().__init__(
            beta, frev, display_units=style(display_units, f="Hz" if log else "kHz")
        )

        self.fmax = fmax
        self.scaling = scaling

        # Create plot axes
        if ax is None:
            _, ax = plt.subplots(**subplots_kwargs)
        self.ax = ax
        self.fig = self.ax.figure

        # Create fft plots
        kwargs = style(plot_kwargs, lw=1)
        (self.artist_plot,) = self.ax.plot([], [], **kwargs)
        self.ax.set(
            xlabel="Frequency " + self.label_for("f"),
            xlim=(0, self.fmax * self.factor_for("f")),
        )
        if log:
            self.ax.set(
                xlim=(10 * self.factor_for("f"), self.fmax * self.factor_for("f")),
                xscale="log",
                yscale="log",
            )
        self.ax.grid()

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
        times = self.factor_for("t") * self.time(particles, mask=mask)

        # re-sample times into equally binned time series
        # optimized to run in < 1s (including fft) even for 1e6 times and fmax = 10 MHz
        dt = 1 / 2 / self.fmax
        n = int(np.ceil((np.max(times) - np.min(times)) / dt))
        # to improve FFT performance, round up to next power of 2
        n = 1 << (n - 1).bit_length()
        dt = (np.max(times) - np.min(times)) / n
        # count timestamps in bins (much faster than np.histogram)
        timeseries = np.bincount(((times - np.min(times)) / dt).astype(int))

        # calculate fft without DC component
        freq = np.fft.rfftfreq(len(timeseries), d=dt)[1:]
        mag = np.abs(np.fft.rfft(timeseries))[1:]
        if self.scaling.lower() == "amplitude":
            # amplitude in units of particle counts
            self.ax.set(ylabel="FFT amplitude")
            mag *= 2 / len(timeseries)
        elif self.scaling.lower() == "pds":
            # power density spectrum in a.u.
            self.ax.set(ylabel="$|\\mathrm{FFT}|^2$")
            mag = mag**2

        # update plot
        self.artist_plot.set_data(freq * self.factor_for("f"), mag)

        if autoscale:
            self.ax.relim()
            self.ax.autoscale()
            self.ax.set(
                xlim=(10 * self.factor_for("f"), self.fmax * self.factor_for("f"))
            )
