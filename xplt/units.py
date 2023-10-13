#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for unit handling

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2023-02-03"


import types
import numpy as np
import pint
import re
from dataclasses import dataclass


@dataclass
class Prop:
    """Class holding generic property information

    Args:
        symbol: Short physical symbol representing property. Preferably latex, e.g. $x$.
        unit: Physical unit of property data.
        description (optional): Longer description of the property to display on legend and axes labels.
    """

    symbol: str
    unit: str
    description: str = None

    def __post_init__(self):
        pint.Unit(self.unit)  # to raise an error if not a valid unit

    def label_for_legend(self):
        """Return label suited as legend entry"""
        return self.description or self.symbol

    def label_for_axes(self, description=True):
        """Return label suited as axes label

        Args:
            description (bool): Whether to include the description (if any)
        """
        label = self.symbol
        if description and self.description:
            label = self.description + "   " + label
        return label

    @staticmethod
    def get(p, custom_properties=None):
        """Create Prop from string

        Args:
            p (str): String to parse. Should be the key of the property to plot.
                May optionally contain a wrapping function call with the property as first argument
                such as `smooth(key)` or `smooth(key, n=10)`.
            custom_properties (dict | None): Dict with custom properties
                to supersede user and default properties.

        Returns:
            Prop: Property information

        Raises:
            ValueError: If property is not known
        """
        if custom_properties and p in custom_properties:
            return custom_properties[p]
        elif p in user_properties:
            return user_properties[p]
        elif p in default_properties:
            return default_properties[p]
        else:
            raise ValueError(
                f"Property `{p}` is not known, please register it using xplt.register_property"
            )


# fmt: off
default_properties = dict(
    
    ## particles
    #############################################################################
    at_turn = Prop("$n$",          unit="1",    description="Turn"          ),  # Turn count
    s       = Prop("$s$",          unit="m",                                ),  # Reference accumulated path length
    x       = Prop("$x$",          unit="m",                                ),  # Horizontal position
    px      = Prop("$x'$",         unit="1",                                ),  # Px / (m/m0 * p0c) = beta_x gamma /(beta0 gamma0)
    y       = Prop("$y$",          unit='m',                                ),  # Vertical position
    py      = Prop("$y'$",         unit="1",                                ),  # Py / (m/m0 * p0c)
    delta   = Prop("$\\delta$",    unit="1",                                ),  # (Pc m0/m - p0c) /p0c
    ptau    = Prop("$p_\\tau$",    unit="1",                                ),  # (Energy m0/m - Energy0) / p0c
    pzeta   = Prop("$p_\\zeta$",   unit="1",                                ),  # ptau / beta0
    #rvv                                                                     # beta / beta0
    #rpp                                                                     # m/m0 P0c / Pc = 1/(1+delta)
    zeta    = Prop("$\\zeta$",     unit='m',                                ),  # (s - beta0 c t )
    tau     = Prop("$\\tau$",      unit='m',                                ),  # (s / beta0 - ct)
    energy  = Prop("$E$",          unit='eV',   description="Total energy"  ),  # Energy (total energy E = sqrt(mc^2 + pc^2))
    chi     = Prop("$\\chi$",      unit="1",                                ),  # q/ q0 * m0/m = qratio / mratio
    
    #mass0   = Prop("m_\\mathrm{ref}",         unit='eV',                    ),   # Reference rest mass
    q0      = Prop("q_\\mathrm{ref}",         unit='e',                     ),   # Reference charge
    #p0c     = Prop("p_\\mathrm{ref}c",        unit='eV',                    ),   # Reference momentum
    #energy0 = Prop("E_\\mathrm{ref}",         unit='eV',                    ),   # Reference energy (total energy)
    #gamma0  = Prop("\\gamma_\\mathrm{ref}",   unit="1",                     ),   # Reference relativistic gamma
    #beta0   = Prop("\\beta_\\mathrm{ref}",    unit="1",                     ),   # Reference relativistic beta
    
    
    ## twiss
    #############################################################################
    betx    = Prop("$\\beta_x$",         unit='m',                          ),  # Horizontal twiss beta-function
    bety    = Prop("$\\beta_y$",         unit='m',                          ),  # Vertical twiss beta-function
    alfx    = Prop("$\\alpha_x$",        unit="1",                          ),  # Horizontal twiss alpha-function
    alfy    = Prop("$\\alpha_y$",        unit="1",                          ),  # Vertical twiss alpha-function
    gamx    = Prop("$\\gamma_x$",        unit='1/m',                        ),  # Horizontal twiss gamma-function
    gamy    = Prop("$\\gamma_y$",        unit='1/m',                        ),  # Vertical twiss gamma-function
    mux     = Prop("$\\mu_x$",           unit="1",                          ),  # Horizontal phase advance
    muy     = Prop("$\\mu_y$",           unit="1",                          ),  # Vertical phase advance
    #muzeta                                                                  #
    qx      = Prop("$q_x$",              unit="1",                          ),  # Horizontal tune qx=mux[-1]
    qy      = Prop("$q_y$",              unit="1",                          ),  # Vertical tune qy=mux[-1]
    qs      = Prop("$q_s$",              unit="1",                          ),  # Synchrotron tune
    dx      = Prop("$D_x$",              unit='m',                          ),  # Horizontal dispersion $D_{x,y}$ [m]
    dy      = Prop("$D_y$",              unit='m',                          ),  # Vertical dispersion $D_{x,y}$ [m]
    #dzeta                                                                   #
    dpx     = Prop("$D_{x'}$",           unit="1",                          ),  # 
    dpy     = Prop("$D_{y'}$",           unit="1",                          ),  # 
    T_rev   = Prop("$T_\\mathrm{rev}$",  unit='s',                          ),  # Revolution period
    f_rev   = Prop("$f_\\mathrm{rev}$",  unit='Hz',                         ),  # Revolution frequency
    circumference               = Prop("$s_\\mathrm{max}$",  unit='m',      ),  # Machine circumference
    slip_factor                 = Prop("$\\eta$",            unit="1",      ),  # eta
    momentum_compaction_factor  = Prop("$\\alpha_c$",        unit="1",      ),  # alpha_c = eta+1/gamma0^2 = 1/gamma0_tr^2
    #betz0                                                                   #
    
    
    ## survey
    #############################################################################
    X        = Prop("$X$",        unit="m",                                 ),  #
    Y        = Prop("$Y$",        unit="m",                                 ),  #
    Z        = Prop("$Z$",        unit="m",                                 ),  #
    theta    = Prop("$\\Theta$",  unit="rad",                               ),  #
    phi      = Prop("$\\Phi$",    unit="rad",                               ),  #
    psi      = Prop("$\\Psi$",    unit="rad",                               ),  #
    
    
    ## beam monitors
    #############################################################################
    count   = Prop("$N$",                   unit="1", description="Count"   ),  # Beam monitor: particle count
    x_mean  = Prop("$\\langle x \\rangle$", unit="m",                       ),  # Beam monitor: average x position
    y_mean  = Prop("$\\langle y \\rangle$", unit="m",                       ),  # Beam monitor: average y position
    x_std   = Prop("$\\sigma_x$",           unit="m",                       ),  # Beam monitor: std of x positions
    y_std   = Prop("$\\sigma_y$",           unit="m",                       ),  # Beam monitor: std of y positions
    
    
    ## derived quantities
    #############################################################################
    t       = Prop("t",           unit='s',                                 ),  # time
    f       = Prop("f",           unit='Hz',                                ),
    # frequency
    
    
)
# fmt: on


user_properties = {}


def register_property(key, unit, symbol=None, description=None):
    """Register a user defined property unit

    Args:
        key (str): Property key as used in `kind` string
        unit (str): Unit of data values associated with this property
        symbol (str, optional): Symbol to display in plots, e.g. $a_1$
        description (str, optional): Description
    """
    user_properties[key] = Prop(symbol or key, unit, description)


@dataclass
class PropToPlot(Prop):
    """Class holding specific property information about something to be plotted

    Args:
        symbol: Short physical symbol representing property. Preferably latex, e.g. $x$.
        unit: Physical unit of property data.
        description (optional): Longer description of the property to display on legend and axes labels.
        key: The property key
        expression (optional): Expression containing some function(s) to evaluate such as `smooth(key,n=10)`.
    """

    key: str = None
    expression: str = None

    def __post_init__(self):
        super().__post_init__()
        if self.key is None:
            raise TypeError("Property key must be set")

    # def __eq__(self, other):
    #    return self.key == other

    # def __str__(self):
    #    return self.key

    @staticmethod
    def get(p, custom_properties=None):
        """Create PropToPlot from string

        Args:
            p (str): String to parse. Should be the key of the property to plot.
                May optionally contain a wrapping function call with the property as first argument
                such as `smooth(key)` or `smooth(key, n=10)`.
            custom_properties (dict | None): Dict with custom properties
                to supersede user and default properties.

        Returns:
            Prop: Property information

        Raises:
            ValueError: If property is not known
        """

        expression = None
        if m := re.match(r".*\(([^(),\s]+)[^()]*\)", p):
            expression = p
            p = m.groups()[0]

        prop = Prop.get(p, custom_properties)

        return PropToPlot(**prop.__dict__, key=p, expression=expression)

    def evaluate_expression(self, data):
        """Evaluate the associated expression for the data"""
        if self.expression:
            from .util import smooth, average

            methods = locals()
            try:
                return eval(self.expression, methods, {self.key: data})
            except Exception as e:
                import inspect, sys

                print(f"Error evaluating exprssion `{self.expression}`", file=sys.stderr)
                print(f"Reason: {e}", file=sys.stderr)
                print(f"Supported methods:", file=sys.stderr)
                for k, v in methods.items():
                    if k in ("self", "data") or k.startswith("_"):
                        continue
                    print(f"  {k}{inspect.signature(v)}", file=sys.stderr)
                raise
        return data

    def label_for_legend(self):
        if self.expression:
            ...  # TODO: expression
        return super().label_for_legend()

    def label_for_axes(self, description=True):
        if self.expression:
            ...  # TODO: expression
        return super().label_for_axes()


## Restrict star imports to local namespace
__all__ = [
    name
    for name, thing in globals().items()
    if not (name.startswith("_") or isinstance(thing, types.ModuleType))
]
