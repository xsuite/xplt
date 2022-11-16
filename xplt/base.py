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
    return dict(default_style, **(kwargs or {}))


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
        
    )   
    # fmt: on

    if p not in units:
        raise NotImplementedError(f"Data unit for parameter {p} not implemented")
    return units.get(p)


def factor_for(var, to_unit):
    """Return factor to convert parameter into unit"""
    if var in ("X", "Y"):
        xy = var.lower()[-1]
        quantity = pint.Quantity(f"({data_unit(xy)})/({data_unit('bet'+xy)})^(1/2)")
    elif var in ("Px", "Py"):
        xy = var[-1]
        quantity = pint.Quantity(f"({data_unit('p'+xy)})*({data_unit('bet'+xy)})^(1/2)")
    else:
        quantity = pint.Quantity(data_unit(var))
    return (quantity / pint.Quantity(to_unit)).to("").magnitude


class XsuitePlot:
    def __init__(
        self,
        display_units=None,
    ):
        """
        Base class for plotting

        :param display_units: Dictionary with units for parameters. Supports prefix notation, e.g. 'bet' for 'betx' and 'bety'.
        """

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

    def factor_for(self, p):
        """Return factor to convert parameter into display unit"""
        return factor_for(p, self.display_unit_for(p))

    def display_unit_for(self, p):
        """Return display unit for parameter"""
        prefix = p[:-1] if len(p) > 1 and p[-1] in "xy" else p
        if p in self._display_units:
            return self._display_units[p]
        if prefix in self._display_units:
            return self._display_units[prefix]
        return data_unit(p)

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


class FixedLimits:
    """Context manager for keeping axis limits fixed while plotting"""

    def __init__(self, axis):
        self.axis = axis

    def __enter__(self):
        self.limits = self.axis.get_xlim(), self.axis.get_ylim()

    def __exit__(self, *args):
        self.axis.set(xlim=self.limits[0], ylim=self.limits[1])
