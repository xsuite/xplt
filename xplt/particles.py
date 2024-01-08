#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting phase space distributions

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-12-07"


import types

import numpy as np
import pint

from .base import XManifoldPlot
from .properties import Property, DerivedProperty, find_property
from .util import c0, get, val, defaults, normalized_coordinates, ieee_mod, defaults_for


class ParticlePlotMixin:
    """Mixin for plotting of particle data

    .. automethod:: _init_particle_mixin
    """

    def _init_particle_mixin(
        self, *, twiss=None, beta=None, frev=None, circumference=None, **kwargs
    ):
        r"""Initializes the mixin by providing associated information

        For a given particle object with inherent properties like ``x``, ``y``, ``px``, ``py``, ``zeta``, ``delta``, etc.
        this mixin allows determination of the following derived properties:

        - Wrapped longitudinal coordinate: ``zeta_wrapped``
           |  This is the ``zeta`` coordinate wrapped to be in range (-circumference/2; circumference/2)
        - Normalized coordinates: ``X``, ``Y``, ``Px``, ``Py``
           |  :math:`X = x/\sqrt{\beta_x} = \sqrt{2J_x} \cos(\Theta_x)`
           |  :math:`P_x = (\alpha_x x + \beta_x p_x)/\sqrt{\beta_x} = -\sqrt{2J_x} \sin(\Theta_x)`
        - Action-angle coordinates: ``Jx``, ``Jy``, ``Θx``, ``Θy``
           |  :math:`J_x = (X^2+P_x^2)/2`
           |  :math:`\Theta_x = -\mathrm{atan2}(P_x, X)`
        - Particle arrival time: ``t``
           |  t = at_turn / frev - zeta / beta / c0

        Prefix notation is also available, i.e. ``P`` for ``Px+Py`` or ``Θ`` for ``Θx+Θy``


        Args:
            twiss (dict | None): Twiss parameters (alfx, alfy, betx and bety) to use for conversion to normalized phase space coordinates.
            beta (float | None): Relativistic beta of particles. Defaults to `beta0` property of particles.
            frev (float | None): Revolution frequency of circular line for calculation of particle time.
            circumference (float | None): Path length of circular line if frev is not given.
            kwargs: Keyword arguments for :class:`~.base.XPlot`

        Returns:
            Updated keyword arguments for :class:`~.base.XPlot` constructor.

        """
        self.twiss = twiss
        self._beta = val(beta)
        self._frev = val(frev)
        self._circumference = val(circumference)

        # Particle specific properties
        # fmt: off
        self._derived_particle_properties = dict(
            zeta_wrapped=DerivedProperty("$\\zeta$", "m", lambda zeta: ieee_mod(zeta, self.circumference)),
            X=DerivedProperty("$X$", "m^(1/2)", lambda x, px, delta: normalized_coordinates(x, px, self.twiss, "x", delta)[0]),
            Y=DerivedProperty("$Y$", "m^(1/2)", lambda y, py, delta: normalized_coordinates(y, py, self.twiss, "y", delta)[0]),
            Px=DerivedProperty("$X'$", "m^(1/2)", lambda x, px, delta: normalized_coordinates(x, px, self.twiss, "x", delta)[1]),
            Py=DerivedProperty("$Y'$", "m^(1/2)", lambda y, py, delta: normalized_coordinates(y, py, self.twiss, "y", delta)[1]),
            Jx=DerivedProperty("$J_x$", "m", lambda X, Px: (X**2 + Px**2) / 2),
            Jy=DerivedProperty("$J_y$", "m", lambda Y, Py: (Y**2 + Py**2) / 2),
            Θx=DerivedProperty("$Θ_x$", "rad", lambda X, Px: -np.arctan2(Px, X)),
            Θy=DerivedProperty("$Θ_y$", "rad", lambda Y, Py: -np.arctan2(Py, Y)),
            t=DerivedProperty("$t$", "s", lambda _data, at_turn, zeta: self._particle_time(at_turn, zeta, _data))
        )
        # fmt: on

        # Update kwargs with particle specific settings
        kwargs["_properties"] = defaults(
            kwargs.get("_properties"), **self._derived_particle_properties
        )
        kwargs["display_units"] = defaults(
            kwargs.get("display_units"),
            X="mm^(1/2)",
            Y="mm^(1/2)",
            P="mm^(1/2)",
            J="mm",  # Action
            Θ="rad",  # Angle
        )

        return kwargs

    @property
    def circumference(self):
        """Circumference of circular accelerator"""
        if self._circumference is not None:
            return self._circumference
        if self.twiss is not None:
            return self.twiss.circumference

    def beta(self, particles=None):
        """Get reference relativistic beta as float"""
        if self._beta is not None:
            return self._beta
        if self.circumference is not None:
            if self._frev is not None:
                return self._frev * self.circumference / c0
            if self.twiss is not None:
                return self.circumference / self.twiss.T_rev0 / c0
        if particles is not None:
            try:
                beta = get(particles, "beta0")
                if np.size(beta) > 1:
                    mean_beta = np.mean(beta)
                    if not np.allclose(beta, mean_beta):
                        raise ValueError(
                            "Particle beta0 is not constant. Please specify beta in constructor!"
                        )
                    beta = mean_beta
                return beta
            except:
                pass

    def frev(self, particles=None):
        """Get reference revolution frequency"""
        if self._frev is not None:
            return self._frev
        if self.twiss is not None:
            return 1 / self.twiss.T_rev0
        beta = self.beta(particles)
        if beta is not None and self.circumference is not None:
            return beta * c0 / self.circumference

    def _particle_time(self, turn, zeta, particles=None):
        """Particle arrival time (t = at_turn / frev - zeta / beta / c0)"""

        # use time directly (if available)
        if particles is not None:
            try:
                return get(particles, "t")
            except AttributeError:
                pass

        # determine time from longitudinal coordinates
        beta = self.beta(particles)
        if beta is None:
            raise ValueError(
                "Particle arrival time can not be determined "
                "because all of the following are unknown: "
                "beta, (frev and circumference), twiss. "
                "To resolve this error, pass either to the plot constructor "
                "or specify particle.beta0."
            )

        time = -zeta / beta / c0  # zeta>0 means early; zeta<0 means late
        if np.any(turn != 0):
            frev = self.frev(particles)
            if frev is None:
                raise ValueError(
                    "Particle arrival time can not be determined while at_turn > 0 "
                    "because all of the following are unknown: "
                    "frev, twiss, (bata and circumference). "
                    "To resolve this error, pass either to the plot constructor "
                    "and/or specify particle.beta0."
                )
            time = time + turn / frev

        return np.array(time)

    def get_property(self, name):
        # Note: this method is not used by the library, but it's handy for standalone use
        """Public method to get a particle property by key

        Args:
            name (str): Key
        Returns:
            Property: The property
        """
        prop = find_property(name, extra_default_properties=self._derived_particle_properties)
        return prop.with_property_resolver(self.get_property)


class ParticleHistogramPlotMixin:
    """Mixin for plotting of histogram particle data

    .. automethod:: _init_particle_histogram_mixin
    """

    def _init_particle_histogram_mixin(self, **kwargs):
        r"""Initializes the mixin by providing associated information

        Args:
            kwargs: Keyword arguments for :class:`~.base.XPlot`

        Returns:
            Updated keyword arguments for :class:`~.base.XPlot` constructor.

        """

        self._histogram_particle_properties = dict(
            count=Property("$N$", "1", description="Particles per bin"),
            cumulative=Property("$N$", "1", description="Particles (cumulative)"),
            rate=Property("$\\dot{N}$", "1/s", description="Particle rate"),
            charge=Property("$Q$", find_property("q").unit, description="Charge per bin"),
            current=Property("$I$", f"({find_property('q').unit})/s", description="Current"),
        )

        # Update kwargs with particle specific settings
        kwargs["_properties"] = defaults(
            kwargs.get("_properties"), **self._histogram_particle_properties
        )
        kwargs["display_units"] = defaults(
            kwargs.get("display_units"),
            current="nA",
        )

        return kwargs

    def _count_based(self, key):
        return key in self._histogram_particle_properties


class ParticlesPlot(XManifoldPlot, ParticlePlotMixin):
    """A plot of particle properties as function of another property"""

    def __init__(
        self,
        particles=None,
        kind="x+y",
        as_function_of="at_turn",
        *,
        mask=None,
        plot_kwargs=None,
        sort_by=None,
        **kwargs,
    ):
        """

        Args:
            particles (Any): Particles data to plot.
            kind (str | list): Defines the properties to plot.
                 This can be a separated string or a nested list or a mixture of both where
                 the first list level (or separator ``,``) determines the subplots,
                 the second list level (or separator ``-``) determines any twinx-axes,
                 and the third list level (or separator ``+``) determines plots on the same axis.
            as_function_of (str): The property to plot as function of.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            plot_kwargs (dict): Keyword arguments passed to the plot function.
            sort_by (str | None): Sort the data by this property. Default is to sort by the ``as_function_of`` property.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments

        """
        kwargs = self._init_particle_mixin(**kwargs)
        kwargs["display_units"] = defaults(kwargs.get("display_units"), bet="m", d="m")
        super().__init__(
            on_x=as_function_of, on_y=kind, on_y_subs={"J": "Jx+Jy", "Θ": "Θx+Θy"}, **kwargs
        )

        # parse kind string
        self.sort_by = sort_by

        # Format plot axes
        self.axis(-1).set(xlabel=self.label_for(self.on_x))

        # create plot elements
        def create_artists(i, j, k, a, p):
            kwargs = defaults_for(
                "plot", plot_kwargs, marker=".", ls="", label=self._legend_label_for((i, j, k))
            )
            return a.plot([], [], **kwargs)[0]

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
            List of artists that have been updated.
        """

        xdata = self.prop(self.on_x).values(
            particles, mask, unit=self.display_unit_for(self.on_x)
        )
        order = np.argsort(
            xdata if self.sort_by is None else self.prop(self.sort_by).values(particles, mask)
        )
        xdata = xdata[order]

        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):
                    values = self.prop(p).values(particles, mask, unit=self.display_unit_for(p))
                    values = values[order]
                    self.artists[i][j][k].set_data((xdata, values))
                    changed.append(self.artists[i][j][k])

                if autoscale:
                    a = self.axis(i, j)
                    a.relim()
                    a.autoscale()

        return changed


## Restrict star imports to local namespace
__all__ = [
    name
    for name, thing in globals().items()
    if not (name.startswith("_") or isinstance(thing, types.ModuleType))
]
