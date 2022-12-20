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
from .util import c0, get, val, defaults, normalized_coordinates
from .base import XPlot, config


class XParticlePlot(XPlot):
    def __init__(
        self,
        *,
        data_units=None,
        display_units=None,
        twiss=None,
        beta=None,
        frev=None,
        circumference=None,
        wrap_zeta=False,
        **kwargs,
    ):
        r"""Base plotting class for particle data

        In addition to the inherent particle properties (like ``x``, ``y``, ``px``, ``py``, ``zeta``, ``delta``, ...)
        the following derived properties are supported (but may require passing of twiss etc.):

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
            data_units (dict, optional): Units of the data. If None, the units are determined from the data.
            display_units (dict, optional): Units to display the data in. If None, the units are determined from the data.
            twiss (dict, optional): Twiss parameters (alfx, alfy, betx and bety) to use for conversion to normalized phase space coordinates.
            beta (float, optional): Relativistic beta of particles. Defaults to particles.beta0.
            frev (float, optional): Revolution frequency of circular line for calculation of particle time.
            circumference (float, optional): Path length of circular line if frev is not given.
            wrap_zeta: If set, wrap the zeta-coordinate plotted at the machine circumference. Either pass the circumference directly or set this to True to use the circumference from twiss.
        """
        display_units = defaults(
            display_units,
            X="mm^(1/2)",
            Y="mm^(1/2)",
            P="mm^(1/2)",
            J="mm",  # Action
            Θ="rad",  # Angle
        )
        prefix_suffix_config = {
            "J": ("Jx", "Jy"),
            "Θ": ("Θx", "Θy"),
        }
        if not config.use_xprime_labels:
            prefix_suffix_config["P"] = ("Px", "Py")
        super().__init__(
            data_units=data_units,
            display_units=display_units,
            prefix_suffix_config=prefix_suffix_config,
        )
        self.twiss = twiss
        self._beta = val(beta)
        self._frev = val(frev)
        self._circumference = val(circumference)
        self.wrap_zeta = wrap_zeta

    @property
    def circumference(self):
        if self._circumference is not None:
            return self._circumference
        if self.twiss is not None:
            return self.twiss.circumference

    def beta(self, particles=None):
        """Get reference relativistic beta as float"""
        if self._beta is not None:
            return self._beta
        if self._frev is not None and self.circumference is not None:
            return self._frev * self.circumference / c0
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
        beta = self.beta(particles)
        if beta is not None and self.circumference is not None:
            return beta * c0 / self.circumference

    def _get_masked(self, particles, prop, mask=None):
        """Get masked particle property"""

        if prop in ("X", "Px", "Y", "Py"):
            # normalized coordinates
            if self.twiss is None:
                raise ValueError("Normalized coordinates requested but twiss is None")
            xy = prop.lower()[-1]
            coords = [self._get_masked(particles, p, mask) for p in (xy, "p" + xy)]
            delta = self._get_masked(particles, "delta", mask)
            X, Px = normalized_coordinates(*coords, self.twiss, xy, delta=delta)
            return X if prop.lower() == xy else Px

        if prop in ("Jx", "Jy", "Θx", "Θy"):
            # action angle coordinates
            X = self._get_masked(particles, prop[-1].upper(), mask)
            Px = self._get_masked(particles, "P" + prop[-1].lower(), mask)
            if prop in ("Jx", "Jy"):  # Action
                return (X**2 + Px**2) / 2
            if prop in ("Θx", "Θy"):  # Angle
                return -np.arctan2(Px, X)

        if prop == "t":
            # particle arrival time (t = at_turn / frev - zeta / beta / c0)
            beta = self.beta(particles)
            if beta is None:
                raise ValueError(
                    "Particle arrival time requested, but enither beta nor beta0 is known. "
                    "Either pass beta or pass both frev and circumference."
                )
            turn = get(particles, "at_turn")[mask]
            zeta = get(particles, "zeta")[
                mask
            ]  # do not use _get_masked as wrap_zeta might mess it up!
            time = -zeta / beta / c0  # zeta>0 means early; zeta<0 means late
            if np.any(turn != 0):
                frev = self.frev(particles)
                if frev is None:
                    raise ValueError(
                        "Particle arrival time requested while at_turn > 0, but neither frev is set, "
                        "nor is circumference and beta known."
                    )
                time = time + turn / frev
            return np.array(time).flatten()

        # default
        v = get(particles, prop)

        if mask is not None:
            v = v[mask]

        if prop == "zeta" and self.wrap_zeta:
            # wrap values at machine circumference
            w = self.circumference if self.wrap_zeta is True else self.wrap_zeta
            v = np.mod(v + w / 2, w) - w / 2

        return np.array(v).flatten()

    def factor_for(self, p):
        """Return factor to convert parameter into display unit"""
        data_unit = None

        if p in ("X", "Y"):  # normalized coordinates
            xy = p.lower()
            data_unit = f"({self.data_unit_for(xy)})/({self.data_unit_for('bet'+xy)})^(1/2)"

        elif p in ("Px", "Py"):  # normalized coordinates
            xy = p[-1]
            data_unit = f"({self.data_unit_for('p'+xy)})*({self.data_unit_for('bet'+xy)})^(1/2)"

        elif p in ("Jx", "Jy"):  # Action
            xy = p[-1]
            data_unit = f"({self.data_unit_for(xy)})^2/({self.data_unit_for('bet'+xy)})"

        elif p in ("Θx", "Θy"):  # Angle
            data_unit = "rad"

        if data_unit is not None:
            quantity = pint.Quantity(data_unit)
            return (quantity / pint.Quantity(self.display_unit_for(p))).to("").magnitude

        return super().factor_for(p)

    def _texify_label(self, label, suffixes=()):
        label = {
            "energy": "E",  # total energy
            "ptau": "p_\\tau",
            "pzeta": "p_\\zeta",
            "chi": "\\chi",
            "energy0": "E_\\mathrm{ref}",  # total energy of reference particle
            "mass0": "m_\\mathrm{ref}",  # mass of reference particle
            "q0": "q_\\mathrm{ref}",  # charge of reference particle
            "p0c": "p_\\mathrm{ref}c",  # momentum of reference particle
            "gamma0": "\\gamma_\\mathrm{ref}",  # relativistic gamma of reference particle
            "beta0": "\\beta_\\mathrm{ref}",  # relativistic beta of reference particle
        }.get(label, label)
        return super()._texify_label(label, suffixes)


class ParticlesPlot(XParticlePlot):
    def __init__(
        self,
        particles=None,
        kind="x+y",
        as_function_of="at_turn",
        *,
        mask=None,
        plot_kwargs=None,
        sort_by=None,
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
        A plot of particle properties as function of another property.

        Args:
            particles: Particles data to plot.
            kind (str or list): Defines the properties to plot.
                 This can be a separated string or a nested list or a mixture of both where
                 the first list level (or separator ``,``) determines the subplots,
                 the second list level (or separator ``-``) determines any twinx-axes,
                 and the third list level (or separator ``+``) determines plots on the same axis.
            as_function_of (str): The property to plot as function of.
            mask: An index mask to select particles to plot. If None, all particles are plotted.
            plot_kwargs (dict): Keyword arguments passed to the plot function.
            sort_by (str or None): Sort the data by this property. Default is to sort by the ``as_function_of`` property.
            grid (bool): If True, show grid lines.
            ax: Axes to plot on. If None, a new figure is created.
            data_units (dict): Units of the data. If None, the units are determined from the data.
            display_units (dict): Units to display the data in. If None, the units are determined from the data.
            twiss (dict): Twiss parameters (alfx, alfy, betx and bety) to use for conversion to normalized phase space coordinates.
            beta (float): Relativistic beta of particles. Defaults to particles.beta0.
            frev (float): Revolution frequency of circular line for calculation of particle time.
            circumference (float): Path length of circular line if frev is not given. Defaults to twiss.circumference.
            wrap_zeta (bool or float): If set, wrap the zeta-coordinate plotted at the machine circumference. Set to
                True to wrap at the circumference or to a value to wrap at this value.
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
        subs = {p: "+".join(s) for p, s in self._prefix_suffix_config.items()}
        self.kind = self._parse_nested_list_string(kind, subs=subs)
        self.as_function_of = as_function_of
        self.sort_by = sort_by

        # initialize figure with n subplots
        nntwins = [len(tw) - 1 for tw in self.kind]
        self._init_axes(ax, len(self.kind), 1, nntwins, grid, sharex="col", **subplots_kwargs)

        # Format plot axes
        self.axis_for(-1).set(xlabel=self.label_for(self.as_function_of))

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

        xdata = self._get_masked(particles, self.as_function_of, mask)
        order = np.argsort(
            xdata if self.sort_by is None else self._get_masked(particles, self.sort_by, mask)
        )
        xdata = xdata[order] * self.factor_for(self.as_function_of)

        changed = []
        for i, ppp in enumerate(self.kind):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):
                    values = self._get_masked(particles, p, mask)
                    values = values[order] * self.factor_for(p)
                    self.artists[i][j][k].set_data((xdata, values))
                    changed.append(self.artists[i][j][k])

                if autoscale:
                    a = self.axis_for(i, j)
                    a.relim()
                    a.autoscale()

        return changed


## Restrict star imports to local namespace
__all__ = [
    name
    for name, thing in globals().items()
    if not (name.startswith("_") or isinstance(thing, types.ModuleType))
]
