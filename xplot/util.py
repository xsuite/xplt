#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Utility methods for plotting

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-08"


import pint


def data_unit(p):
    """Return data unit of parameter p as used by xsuite"""
    # https://github.com/xsuite/xsuite/raw/main/docs/physics_manual/physics_man.pdf
    # fmt: off
    units = dict(        
        
        ## particles
        ###################
        s='m',            # Reference accumulated path length
        x='m',            # Horizontal position
        px=1,             # Px / (m/m0 * p0c) = beta_x gamma /(beta0 gamma0)
        y='m',            # Vertical position
        py=1,             # Py / (m/m0 * p0c)
        delta=1,          # (Pc m0/m - p0c) /p0c
        ptau=1,           # (Energy m0/m - Energy0) / p0c
        pzeta=1,          # ptau / beta0
        rvv=1,            # beta / beta0
        rpp=1,            # m/m0 P0c / Pc = 1/(1+delta)
        zeta='m',         # (s - beta0 c t )
        tau='m',          # (s / beta0 - ct)
        mass0='eV',       # Reference rest mass
        q0='e',           # Reference charge
        p0c='eV',         # Reference momentum
        energy0='eV',     # Reference energy
        gamma0=1,         # Reference relativistic gamma
        beta0=1,          # Reference relativistic beta
        
        ## twiss
        ###################
        betx='m',         # Horizontal twiss beta-function
        bety='m',         # Vertical twiss beta-function
        alfx=1,           # Horizontal twiss alpha-function
        alfy=1,           # Vertical twiss alpha-function
        gamx='1/m',       # Horizontal twiss gamma-function
        gamy='1/m',       # Vertical twiss gamma-function
        mux=1,            # Horizontal phase advance
        muy=1,            # Vertical phase advance
        #muzeta
        qx=1,             # Horizontal tune qx=mux[-1]
        qy=1,             # Vertical tune qy=mux[-1]
        #qs
        dx='m',           # Horizontal dispersion $D_{x,y}$ [m]
        dy='m',           # $D_{x,y}$ [m]
        #dzeta
        dpx=1,
        dpy=1,
        circumference='m',
        T_rev='s',
        slip_factor=1,                 # eta
        momentum_compaction_factor=1,  # alpha_c = eta+1/gamma0^2 = 1/gamma0_tr^2
        #betz0
        
    )   
    # fmt: on

    if p not in units:
        raise NotImplementedError(f"Data unit for parameter {p} not implemented")
    return units.get(p)


def factor_for(p, unit):
    """Return factor to convert parameter into unit"""
    return (pint.Quantity(data_unit(p)) / pint.Quantity(unit)).to("").magnitude
