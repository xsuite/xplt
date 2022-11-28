#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Base methods for plotting

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-08"


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pint
import re


VOID = object()


def get(obj, val, default=VOID):
    """Get value from object"""
    try:
        return getattr(obj, val)
    except:
        try:
            return obj[val]
        except:
            if default is not VOID:
                return default
    raise AttributeError(f"{obj} does not provide an attribute or index '{val}'")


def style(kwargs, **default_style):
    """Return kwargs or defaults"""
    kwargs = kwargs or {}
    if "c" in kwargs or "color" in kwargs:
        # c and color are common aliases, remove both from default_style if present
        default_style.pop("c", None)
        default_style.pop("color", None)
    return dict(default_style, **kwargs)


def data_unit(p):
    """Return data unit of parameter p as used by xsuite"""
    # https://github.com/xsuite/xsuite/raw/main/docs/physics_manual/physics_man.pdf
    if re.fullmatch(r"k(\d+)l", p):
        return 1 if p == "k0l" else "m^-" + p[1:-1]
    # fmt: off
    units = dict(

        ## particles
        ###################
        s='m',            # Reference accumulated path length
        x='m',            # Horizontal position
        px="1",           # Px / (m/m0 * p0c) = beta_x gamma /(beta0 gamma0)
        y='m',            # Vertical position
        py="1",           # Py / (m/m0 * p0c)
        delta="1",        # (Pc m0/m - p0c) /p0c
        ptau="1",         # (Energy m0/m - Energy0) / p0c
        pzeta="1",        # ptau / beta0
        rvv="1",          # beta / beta0
        rpp="1",          # m/m0 P0c / Pc = 1/(1+delta)
        zeta='m',         # (s - beta0 c t )
        tau='m',          # (s / beta0 - ct)
        mass0='eV',       # Reference rest mass
        q0='e',           # Reference charge
        p0c='eV',         # Reference momentum
        energy0='eV',     # Reference energy
        gamma0="1",       # Reference relativistic gamma
        beta0="1",        # Reference relativistic beta
        
        ## twiss
        ###################
        betx='m',         # Horizontal twiss beta-function
        bety='m',         # Vertical twiss beta-function
        alfx="1",         # Horizontal twiss alpha-function
        alfy="1",         # Vertical twiss alpha-function
        gamx='1/m',       # Horizontal twiss gamma-function
        gamy='1/m',       # Vertical twiss gamma-function
        mux="1",          # Horizontal phase advance
        muy="1",          # Vertical phase advance
        #muzeta
        qx="1",           # Horizontal tune qx=mux[-1]
        qy="1",           # Vertical tune qy=mux[-1]
        #qs
        dx='m',           # Horizontal dispersion $D_{x,y}$ [m]
        dy='m',           # $D_{x,y}$ [m]
        #dzeta
        dpx="1",
        dpy="1",
        circumference='m',
        T_rev='s',
        slip_factor="1",                 # eta
        momentum_compaction_factor="1",  # alpha_c = eta+1/gamma0^2 = 1/gamma0_tr^2
        #betz0

        ## derived quantities
        ######################
        t='s',
        f='Hz',
        
    )   
    # fmt: on

    if p not in units:
        raise NotImplementedError(f"Data unit for parameter {p} not implemented")
    return units.get(p)


class ManifoldMultipleLocator(mpl.ticker.MaxNLocator):
    def __init__(self, fixed_multiples, n=5, minor_n=None):
        """A multiple locator that chooses its base from a set of multiples to yield about n ticks

        For ranges smaller than the smallest fixed_multiple, the default MaxNLocator is used
        For ranges larger than the largest fixed_multiple, a multiple of the later is used

        Args:
            fixed_multiples (list of float): multiples to choose from
            n (int): number of ticks to produce (best effort)
            minor_n (list of float): If given, produce that many minor ticks in between each major tick. Length must match the length of fixed_multiples
        """
        super().__init__(n)
        self.fixed_multiples = fixed_multiples
        self.n = n
        self.minor_n = minor_n

    def _raw_ticks(self, vmin, vmax):
        if vmax - vmin < self.n * self.fixed_multiples[0]:
            return super()._raw_ticks(vmin, vmax)
        for step in self.fixed_multiples:
            if (vmax - vmin) / step <= self.n:
                break
        while (vmax - vmin) / step > self.n:
            step += self.fixed_multiples[-1]
        if self.minor_n is not None:
            if step in self.fixed_multiples:
                step /= self.minor_n[self.fixed_multiples.index(step)]
            else:
                step /= self.minor_n[-1]
        return np.arange(int(vmin / step) * step, vmax + step, step)


class AngleLocator(ManifoldMultipleLocator):
    def __init__(self, minor=False, deg=True):
        """A tick locator for angles

        Args:
            minor (bool): If true, return a minor locator. By default a major locator is returned.
            deg (bool): If true, locate angles is degree. If false, in radians.
        """
        multiples = (5, 15, 30, 45, 60, 90, 120, 180, 360)
        if not deg:
            multiples = list(np.deg2rad(multiples))
        subdivide = (5, 3, 3, 3, 4, 3, 4, 4, 4)
        super().__init__(multiples, 5, subdivide if minor else None)


class RadiansFormatter(mpl.ticker.Formatter):
    """A tick formatter to format angles in radians as fractions or multiples of pi"""

    def __call__(self, x, pos=None):
        if x == 0:
            return "0"
        s = "-" if x < 0 else ""
        x = abs(x)
        if x == np.pi:
            return f"${s}\\pi$"
        for n in (2, 3, 4, 6, 8, 12):
            m = round(x / (np.pi / n))
            if abs(x - m * np.pi / n) < 1e-10 and m / n != m // n:
                if m == 1:
                    m = ""
                return f"${s}{m}\\pi/{n}$"
        return f"${x/np.pi:g}\\pi$"


class Xplot:
    def __init__(
        self,
        *,
        data_units=None,
        display_units=None,
    ):
        """
        Base class for plotting

        Args:
            display_units: Dictionary with units to display parameters. Supports prefix notation, e.g. 'bet' for 'betx' and 'bety'.
            data_units: Dictionary with native units of parameters. Supports prefix notation, e.g. 'bet' for 'betx' and 'bety'.
        """

        self._data_units = data_units or {}
        self._display_units = dict(
            dict(
                x="mm",
                y="mm",
                p="mrad",
                X="mm^(1/2)",
                Y="mm^(1/2)",
                P="mm^(1/2)",
                k0l="rad",
            ),
            **(display_units or {}),
        )

    def save(self, fname, **kwargs):
        """Save the figure"""
        self.fig.savefig(fname, **style(kwargs, dpi=300))

    def factor_for(self, p):
        """Return factor to convert parameter into display unit"""
        if p in ("X", "Y"):
            xy = p.lower()[-1]
            quantity = pint.Quantity(
                f"({self.data_unit_for(xy)})/({self.data_unit_for('bet'+xy)})^(1/2)"
            )
        elif p in ("Px", "Py"):
            xy = p[-1]
            quantity = pint.Quantity(
                f"({self.data_unit_for('p'+xy)})*({self.data_unit_for('bet'+xy)})^(1/2)"
            )
        else:
            quantity = pint.Quantity(self.data_unit_for(p))
        return (quantity / pint.Quantity(self.display_unit_for(p))).to("").magnitude

    def data_unit_for(self, p):
        """Return data unit for parameter"""
        if p in self._data_units:
            return self._data_units[p]
        prefix = p[:-1] if len(p) > 1 and p[-1] in "xy" else p
        if prefix in self._data_units:
            return self._data_units[prefix]
        return data_unit(p)

    def display_unit_for(self, p):
        """Return display unit for parameter"""
        if p in self._display_units:
            return self._display_units[p]
        prefix = p[:-1] if len(p) > 1 and p[-1] in "xy" else p
        if prefix in self._display_units:
            return self._display_units[prefix]
        return self.data_unit_for(p)

    def label_for(self, *pp, unit=True):
        """
        Return label for list of parameters, joining where possible

        :param pp: Parameter names
        :param unit: Wheather to include unit
        """

        def texify(label):
            if m := re.fullmatch(r"k(\d+)l", label):
                return f"k_{m.group(1)}l"
            return {
                "alf": "\\alpha",
                "bet": "\\beta",
                "gam": "\\gamma",
                "mu": "\\mu",
                "d": "D",
            }.get(label, label)

        def split(p):
            if p[-1] in "xy":
                return p[:-1], p[-1]
            return p, ""

        # split pre- and suffix
        prefix, _ = split(pp[0])
        display_unit = self.display_unit_for(pp[0])
        suffix = []
        for p in pp:
            pre, suf = split(p)
            suffix.append(suf)
            if pre != prefix or self.display_unit_for(p) != display_unit:
                # no common prefix or different units, treat separately!
                return " ,  ".join([self.label_for(p) for p in pp])
        suffix = ",".join(suffix)

        # build label
        label = "$"
        if prefix:
            label += texify(prefix)
            if suffix:
                label += "_{" + suffix + "}"
        else:
            label += suffix
        label += "$"
        if unit and display_unit:
            display_unit = pint.Unit(display_unit)
            if display_unit != pint.Unit("1"):
                label += f" / ${display_unit:~l}$"
        return label

    @staticmethod
    def set_axis_ticks_angle(yaxis, minor=True, deg=False):
        """Set ticks locator and formatter to display an angle

        This will set ticks at multiples or fractions of 180° or pi with appropriate labels

        Args:
            yaxis: The axis to format (ax.xaxis or ax.yaxis)
            minor (bool): If true (default), also set the minor locator
            deg (bool): If true, use angles in degree. If false (default), in radians.
        """
        yaxis.set_major_locator(AngleLocator(deg=deg))
        if minor:
            yaxis.set_minor_locator(AngleLocator(deg=deg, minor=True))
        if not deg:
            yaxis.set_major_formatter(RadiansFormatter())

    @staticmethod
    def plot_harmonics(
        ax, v, dv=0, *, n=20, vertical=True, inverse=False, **plot_kwargs
    ):
        """Add vertical lines or spans indicating the location of values or spans and their harmonics

        Args:
            ax: Axis to plot onto.
            v (float or list of float): Value or list of values.
            dv (float or list of float, optional): Width or list of widths centered around value(s).
            n (int): Number of harmonics to plot.
            vertical (bool): Plot vertical lines if true, horizontal otherweise.
            inverse (bool): If true, plot harmonics of n/(v±dv) instead of n*(v±dv). Useful to plot frequency harmonics in time domain and vice-versa.
            plot_kwargs: Keyword arguments to be passed to plotting method
        """
        if not hasattr(v, "__iter__"):
            v = [v]
        if not hasattr(dv, "__iter__"):
            dv = [dv] * len(v)
        kwargs = style(plot_kwargs, zorder=1.9, color="gray", lw=1)
        for i in range(1, n + 1):
            for j, (vi, dvi) in enumerate(zip(v, dv)):
                if dvi == 0:
                    method = ax.axvline if vertical else ax.axhline
                    args = np.array([vi])
                else:
                    method = ax.axvspan if vertical else ax.axhspan
                    args = np.array([vi - dvi / 2, vi + dvi / 2])
                args = sorted((i / args) if inverse else (i * args))
                method(
                    *args,
                    **style(kwargs, alpha=1 - np.log(1 + (np.e - 1) * (i - 1) / n)),
                )
                kwargs.pop("label", None)

    @staticmethod
    def add_scale(
        ax,
        scale,
        label=None,
        *,
        vertical=False,
        width=0.01,
        padding=0.1,
        loc="auto",
        color="k",
        fontsize="x-small",
    ):
        """Add a scale patch (a yardstick or ruler)

        Args:
            ax: The axis to add it to.
            scale: The size of the scale in data units.
            label (str, optional): A label for the scale.
            vertical (bool): If true, make a vertical one (default is a horizontal one).
            width (float): The line width of the scale in axis units.
            padding (float): The padding between the scale and the axis.
            loc (str): The location of the scale. Can be any of the usual matplotlib locations, e.g. 'auto', 'upper left', 'upper center', 'upper right', 'center left', 'center', 'center right', 'lower left', 'lower center, 'lower right'.
            color: Color for the patch.
            fontsize: Font size of the label.

        Returns:
            The artist added (an AnchoredOffsetbox).
        """
        if loc == "auto":
            loc = "upper left" if vertical else "lower right"
        w, h = scale, width
        w_trans, h_trans = ax.transData, ax.transAxes
        if vertical:  # swap dimensions
            w, h = h, w
            w_trans, h_trans = h_trans, w_trans
        aux = mpl.offsetbox.AuxTransformBox(
            mpl.transforms.blended_transform_factory(w_trans, h_trans)
        )
        aux.add_artist(plt.Rectangle((0, 0), w, h, fc=color))
        if label:
            kwa = dict(text=label, color=color, fontsize=fontsize)
            if vertical:
                aux.add_artist(
                    plt.Text(w * 2, h / 2, ha="left", va="center", rotation=90, **kwa)
                )
            else:
                aux.add_artist(
                    plt.Text(w / 2, h * 1.5, va="bottom", ha="center", **kwa)
                )
        ab = mpl.offsetbox.AnchoredOffsetbox(
            loc, borderpad=padding, zorder=100, frameon=False
        )
        ab.set_child(aux)
        ax.add_artist(ab)
        return ab


class FixedLimits:
    """Context manager for keeping axis limits fixed while plotting"""

    def __init__(self, axis):
        self.axis = axis

    def __enter__(self):
        self.limits = self.axis.get_xlim(), self.axis.get_ylim()

    def __exit__(self, *args):
        self.axis.set(xlim=self.limits[0], ylim=self.limits[1])
