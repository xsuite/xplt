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
    if p in "s,x,y,zeta,tau,betx,bety,dx,dy":
        return "m"
    if p in "px,py,delta,alfx,alfy,mux,muy,qx,qy":
        return 1
    ...  # TODO


def factor_for(p, unit):
    """Return factor to convert parameter into unit"""
    return (pint.Quantity(data_unit(p)) / pint.Quantity(unit)).to("").magnitude
