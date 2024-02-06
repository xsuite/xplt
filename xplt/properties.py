#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods to work with properties

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2023-11-11"


import re
import types
import inspect
from typing import Callable, Optional
from dataclasses import dataclass

import numpy as np
import pint

try:
    import pandas as pd
except ImportError:
    # pandas is an optional dependency
    pd = None

from .util import get


arb_unit = "arb. unit"


class Property:
    def __init__(self, symbol, unit, description=None):
        """Class holding generic property information

        Args:
            symbol (str): Short physical symbol representing property, preferably latex (e.g. $x$).
            unit (str): Physical unit of property data.
            description (str | None): Longer description of the property to display on legend and axes labels.
        """
        self.symbol = symbol
        self.unit = unit
        self.description = description

        pint.Unit(self.unit)  # to raise an error if not a valid unit

    def values(self, data, mask=None, *, unit=None):
        """Get masked data for this property

        Args:
            data (any): The data object providing the values
            mask (any): An optional mask to apply to the values
            unit (str): The unit to convert the data to, must be compatible with this property

        """
        raise NotImplementedError("Abstract property does not provide values")

    def prop(self, name):
        """Find a property by name

        This method is overwritten by plots which provide custom properties

        Args:
            name (str): The property name by which it is registered
        Returns:
             Property
        """
        return find_property(name).with_property_resolver(self.prop)

    def with_property_resolver(self, resolver):
        """Create a new property with a custom property resolver

        Args:
            resolver (callable): A function which takes a property name and returns a property
        """
        p = self.__class__(self.symbol, self.unit, self.description)
        p.prop = resolver
        return p

    def __repr__(self):
        # include description only if not None
        r = f"{self.__class__.__name__}({self.symbol}, unit={self.unit}"
        if self.description is not None:
            r += f", description={self.description}"
        return r + ")"


class DataProperty(Property):
    def __init__(self, key, unit, symbol=None, description=None):
        """Class for property which can directly be accessed from data

        Args:
            key (str): The key used to access data (or None if you promise not to call .value)
            unit (str): Physical unit of property data.
            symbol (str | None): Short physical symbol representing property, preferably latex (e.g. $x$).
                                 If None, defaults to the key.
            description (optional): Longer description of the property to display on legend and axes labels.
        """
        super().__init__(symbol or f"${key}$", unit, description)
        self.key = key

    def with_property_resolver(self, resolver):
        p = self.__class__(self.key, self.unit, self.symbol, self.description)
        p.prop = resolver
        return p

    def values(self, data, mask=None, *, unit=None):
        """Get masked data for this property

        Args:
            data (any): The data object providing the values
            mask (None | Any | callable): An optional mask to apply to the values.
                Can be None, a slice, a binary mask or a callback.
                If a callback, it must have the signature ``(mask_1, get) -> mask_2`` where mask_1 is the
                binary mask to be modified, mask_2 is the modified mask, and get is a method allowing the
                callback to retrieve particle properties in their respective data units.
                Example:
                    def mask_callback(mask, get):
                        mask &= get("t") < 1e-3  # all particles with time < 1 ms
                        return mask
            unit (str): The unit to convert the data to, must be compatible with this property
        """

        # try to get requested property from particle data
        if self.key is None:
            raise RuntimeError("Cannot get data from a property with key None")
        v = get(data, self.key)

        # apply mask
        v = np.array(v)
        if v.ndim > 0:
            if callable(mask):
                v = v.flatten()
                mask = mask(
                    np.ones_like(v, dtype="bool"), lambda key: self.prop(key).values(data)
                )
            if mask is not None:
                v = v[mask]

        # flatten
        v = v.flatten()

        # convert to unit
        if unit is not None:
            factor = pint.Quantity(1, self.unit).to(unit).magnitude
            v *= factor

        return v


class DerivedProperty(Property):
    def __init__(self, symbol, unit, evaluate, description=None):
        """Class for property which is derived from other properties

        Args:
            symbol (str | None): Short physical symbol representing property, preferably latex (e.g. $x$).
            unit (str): Physical unit of property data.
            evaluate (Callable): The function which determines the property values. Function parameters must
                                 be names of other properties, which will be provided to the function.
            description (optional): Longer description of the property to display on legend and axes labels.
        """
        super().__init__(symbol, unit, description)
        self.evaluate = evaluate

    def with_property_resolver(self, resolver):
        p = self.__class__(self.symbol, self.unit, self.evaluate, self.description)
        p.prop = resolver
        return p

    def values(self, data, mask=None, *, unit=None):
        """Get masked data for this property

        Args:
            *: See :py:func:`Property.values`
        """

        # determine dependencies
        dependents = inspect.signature(self.evaluate).parameters.keys()

        # collect dependencies
        dv = {}
        for d in dependents:
            if d == "_data":  # special parameter to request raw data object
                dv[d] = data
            else:
                p = self.prop(d)
                dv[d] = p.values(data, mask=mask)

        # evaluate
        v = np.array(self.evaluate(**dv))

        # convert to unit
        if unit is not None:
            factor = pint.Quantity(1, self.unit).to(unit).magnitude
            v *= factor

        return v


_default_properties = {}
_user_properties = {}


def find_property(name, *, extra_user_properties=None, extra_default_properties=None):
    """Find a Property by name

    Args:
        name (str): The name of the property.
        extra_user_properties (dict | None): Additional user properties.
        extra_default_properties (dict | None): Additional default properties.

    Returns:
        Property: Property information

    Raises:
        ValueError: If property is not known
    """
    if extra_user_properties and name in extra_user_properties:
        return extra_user_properties[name]
    elif name in _user_properties:
        return _user_properties[name]
    elif extra_default_properties and name in extra_default_properties:
        return extra_default_properties[name]
    elif name in _default_properties:
        return _default_properties[name]
    else:
        raise ValueError(
            f"Property `{name}` is not known, please register it using `xplt.register_property` or `xplt.register_derived_property`."
        )


def register_property(name, property):
    """Register a user defined property

    Args:
        name (str): Property name as used in `kind` string
        property (Property): The property
    """
    _user_properties[name] = property


def register_data_property(name, data_unit, symbol=None, description=None):
    """Register a user defined data property

    Args:
        name (str): Property name as used in `kind` string
        data_unit (str): Unit of data values associated with this property
        symbol (str, optional): Symbol to display in plots, e.g. $a_1$
        description (str, optional): Description
    """
    register_property(name, DataProperty(name, data_unit, symbol, description))


def register_derived_property(name, function, unit=None, symbol=None, description=None):
    """Register a user defined derived property

    Args:
        name (str): Property name as used in `kind` string
        function (Callable): Function to evaluate the property from other properties
        unit (str | None): Unit of data values associated with this property.
                           If none, the unit is determined from the function return value.
        symbol (str, optional): Symbol to display in plots, e.g. $a_1$
        description (str, optional): Description
    """
    if unit is None:
        # determine unit from function return value
        inputs = inspect.signature(function).parameters.keys()
        inputs = {p: pint.Quantity(1, find_property(p).unit) for p in inputs}
        unit = function(**inputs).units
    register_property(name, DerivedProperty(symbol or f"${name}$", unit, function, description))


# Property definitions
#######################


P = DataProperty


for p in [
    ##
    ## particles
    #############################################################################
    P("at_turn", "1", "$n$", description="Turn"),  # Turn count
    P("s", "m"),  # Reference accumulated path length
    P("x", "m"),  # Horizontal position
    P("px", "1", "$x'$"),  # Px / (m/m0 * p0c) = beta_x gamma /(beta0 gamma0)
    P("y", "m"),  # Vertical position
    P("py", "1", "$y'$"),  # Py / (m/m0 * p0c)
    P("delta", "1", "$\\delta$"),  # (Pc m0/m - p0c) /p0c
    P("zeta", "m", "$\\zeta$"),  # (s - beta0 c t )
    P("tau", "m", "$\\tau$"),  # (s / beta0 - ct)
    P("ptau", "1", "$p_\\tau$"),  # (Energy m0/m - Energy0) / p0c
    P("pzeta", "1", "$p_\\zeta$"),  # ptau / beta0
    P("energy", "eV", "$E$", description="Total energy"),  # E = sqrt(mc^2 + pc^2))
    P("chi", "1", "$\\chi$"),  # q/ q0 * m0/m = qratio / mratio
    P(
        "charge_ratio", "1", "$q/q_\\mathrm{ref}$", description="Charge ratio"
    ),  # charge ratio: q/q0
    P("q0", "e", "$q_\\mathrm{ref}$", description="Reference charge"),
    P("mass0", "eV/c^2", "$m_\\mathrm{ref}$", description="Reference rest mass"),
    P(
        "chi",
        "1",
        "$q/q_\\mathrm{ref} \\cdot m_\\mathrm{ref}/m$",
        description="Charge-over-rest-mass ratio",
    ),  # chi = q/q0 * m0/m
    # P("p0c", "eV", "p_\\mathrm{ref}c", description="Reference momentum"),
    # P("energy0", "eV", "E_\\mathrm{ref}", description="Reference energy (total energy)"),
    # P("gamma0", "1", "\\gamma_\\mathrm{ref}", description="Reference relativistic gamma"),
    # P("beta0", "1", "\\beta_\\mathrm{ref}", description="Reference relativistic beta"),
    ##
    ## twiss
    #############################################################################
    P("betx", "m", "$\\beta_x$"),  # Horizontal twiss beta-function
    P("bety", "m", "$\\beta_y$"),  # Vertical twiss beta-function
    P("alfx", "1", "$\\alpha_x$"),  # Horizontal twiss alpha-function
    P("alfy", "1", "$\\alpha_y$"),  # Vertical twiss alpha-function
    P("gamx", "1/m", "$\\gamma_x$"),  # Horizontal twiss gamma-function
    P("gamy", "1/m", "$\\gamma_y$"),  # Vertical twiss gamma-function
    P("mux", "1", "$\\mu_x$"),  # Horizontal phase advance
    P("muy", "1", "$\\mu_y$"),  # Vertical phase advance
    P("dx", "m", "$D_x$"),  # Horizontal dispersion
    P("dy", "m", "$D_y$"),  # Vertical dispersion
    P("dpx", "1", "$D_{x'}$"),  # Horizontal dispersion of px
    P("dpy", "1", "$D_{y'}$"),  # Vertical dispersion of py
    ##
    ## beam monitors
    #############################################################################
    P("count", "1", "$N$", description="Count"),  # Beam monitor: particle count
    P("x_mean", "m", "$\\langle x \\rangle$"),  # Beam monitor: average x position
    P("y_mean", "m", "$\\langle y \\rangle$"),  # Beam monitor: average y position
    P("x_std", "m", "$\\sigma_x$"),  # Beam monitor: std of x positions
    P("y_std", "m", "$\\sigma_y$"),  # Beam monitor: std of y positions
]:
    _default_properties.update({p.key: p})

_default_properties.update(
    dict(
        q=DerivedProperty(
            "$q$", "e", lambda q0, charge_ratio: q0 * charge_ratio, description="Charge"
        ),
        m=DerivedProperty(
            "$m$",
            "eV/c^2",
            lambda chi, charge_ratio, mass0: charge_ratio / chi * mass0,
            description="Rest mass",
        ),
    )
)


## Restrict star imports to local namespace
__all__ = [
    name
    for name, thing in globals().items()
    if not (name.startswith("_") or isinstance(thing, types.ModuleType))
]
