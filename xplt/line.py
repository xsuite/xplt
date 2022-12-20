#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting lines

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-08"

import types

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import re

from .util import defaults, get
from .base import XPlot


def iter_elements(line):
    """Iterate over elements in line

    Yields (name, element, s_from, s_to) with

    """
    el_s0 = line.get_s_elements("upstream")
    el_s1 = line.get_s_elements("downstream")
    for name, el, s0, s1 in zip(line.element_names, line.elements, el_s0, el_s1):
        if s0 == s1:  # thin lense located at element center
            if hasattr(el, "length"):
                s0, s1 = (s0 + s1 - el.length) / 2, (s0 + s1 + el.length) / 2
        yield name, el, s0, s1


class KnlPlot(XPlot):
    def __init__(
        self,
        line=None,
        *,
        knl=None,
        ax=None,
        filled=True,
        data_units=None,
        display_units=None,
        resolution=1000,
        line_length=None,
        **subplots_kwargs,
    ):
        """
        A plot for knl values along line

        Args:
            line: Line of elements.
            knl (int or list of int): Maximum order or list of orders n to plot knl values for. If None, automatically determine from line.
            ax: An axes to plot onto. If None, a new figure is created.
            filled (bool): If True, make a filled plot instead of a line plot.
            data_units (dict, optional): Units of the data. If None, the units are determined from the data.
            display_units (dict, optional): Units to display the data in. If None, the units are determined from the data.
            resolution: Number of points to use for plot.
            line_length: Length of line (only required if line is None).
            subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.

        Known issues:
            - Thin elements produced with MAD-X MAKETHIN do overlap due to the displacement introduced by the TEAPOT algorithm.
              This leads to glitches of knl being doubled or zero at element overlaps for lines containing such elements.

        """
        super().__init__(display_units=defaults(display_units, k0l="rad"))

        if knl is None:
            if line is None:
                raise ValueError("Either line or knl parameter must not be None")
            knl = range(max([e.order for e in line.elements if hasattr(e, "order")]) + 1)
        if isinstance(knl, int):
            knl = range(knl + 1)
        self.knl = knl
        if line is None and line_length is None:
            raise ValueError("Either line or line_length parameter must not be None")
        self.S = np.linspace(0, line_length or line.get_length(), resolution)
        self.filled = filled

        # Create plot
        if ax is None:
            _, ax = plt.subplots(**subplots_kwargs)
        self.fig, self.ax = ax.figure, ax
        self.ax.set(
            xlabel=self.label_for("s"),
            ylabel="$k_nl$",
        )
        self.ax.grid()

        # create plot elements
        self.artists = []
        for n in self.knl:
            if self.filled:
                artist = self.ax.fill_between(
                    self.S,
                    np.zeros_like(self.S),
                    alpha=0.5,
                    label=self.label_for(f"k{n}l"),
                    zorder=3,
                )
            else:
                (artist,) = self.ax.plot([], [], alpha=0.5, label=self.label_for(f"k{n}l"))
            self.artists.append(artist)
        self.ax.plot(self.S, np.zeros_like(self.S), "k-", lw=1)
        self.ax.legend(ncol=5)

        # set data
        if line is not None:
            self.update(line, autoscale=True)

    def update(self, line, autoscale=False):
        """
        Update the line data this plot shows

        :param line: Line of elements.
        :param autoscale: Whether or not to perform autoscaling on all axes
        :return: changed artists
        """
        # compute knl as function of s
        KNL = np.zeros((len(self.knl), self.S.size))
        Smax = line.get_length()
        for name, el, s0, s1 in iter_elements(line):
            if hasattr(el, "knl"):
                for i, n in enumerate(self.knl):
                    if n <= el.order:
                        if 0 <= s0 <= Smax:
                            mask = (self.S >= s0) & (self.S < s1)
                        else:
                            # handle wrap around
                            mask = (self.S >= s0 % Smax) | (self.S < s1 % Smax)
                        KNL[i, mask] += el.knl[n]

        # plot
        s = self.factor_for("s")
        changed = []
        for n, art, knl in zip(self.knl, self.artists, KNL):
            f = self.factor_for(f"k{n}l")
            if self.filled:
                art.get_paths()[0].vertices[1 : 1 + knl.size, 1] = f * knl
            else:
                art.set_data((s * self.S, f * knl))
            changed.append(art)

        if autoscale:
            if self.filled:  # At present, relim does not support collection instances.
                self.ax.update_datalim(
                    mpl.transforms.Bbox.union(
                        [a.get_datalim(self.ax.transData) for a in self.artists]
                    )
                )
            else:
                self.ax.relim()
            self.ax.autoscale()
            self.ax.set(xlim=(s * self.S.min(), s * self.S.max()))

        return changed

    def _texify_label(self, label, suffixes=()):
        if m := re.fullmatch(r"k(\d+)l", label):
            label = f"k_{m.group(1)}l"
        return super()._texify_label(label, suffixes)


class FloorPlot(XPlot):
    def __init__(
        self,
        survey=None,
        line=None,
        projection="ZX",
        *,
        boxes=None,
        labels=False,
        element_width=1,
        ax=None,
        data_units=None,
        display_units=None,
        **subplots_kwargs,
    ):
        """
        A floor plan of the line based on survey data

        Args:
            survey: Survey data.
            projection: The projection to use: A pair of coordinates ('XZ', 'ZY' etc.)
            line: Line data with additional information about elements.
                Use this to have colored boxes of correct size etc.
            boxes: Config option for showing colored boxes for elements. See below.
                Detailed options can be "length" and all options suitable for a patch,
                such as "color", "alpha", etc.
            labels: Config option for showing labels for elements. See below.
                Detailed options can be "text" (e.g. "Dipole {name}" where name will be
                replaced with the element name) and all options suitable for an annotation,
                such as "color", "alpha", etc.
            element_width (float): Width of element boxes.
            ax: An axes to plot onto. If None, a new figure is created.
            data_units (dict, optional): Units of the data. If None, the units are determined from the data.
            display_units (dict, optional): Units to display the data in. If None, the units are determined from the data.
            subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.

        The config options can be:
            - None: Use good defaults.
            - A bool: En-/disable option for all elements (except drifts).
            - A str (regex): Filter by element name.
            - A dict: Detailed options to apply for each element in the form of
                {"regex": {...}}. For each matching element name, the options are used.

        """

        super().__init__(
            data_units=defaults(
                data_units, X="m", Y="m", Z="m", theta="rad", phi="rad", psi="rad"
            ),
            display_units=display_units,
        )

        if projection == "3D":
            raise NotImplementedError()

        self.projection = projection
        self.boxes = boxes
        self.labels = labels
        self.element_width = element_width

        # Create plot
        if ax is None:
            _, self.ax = plt.subplots(**subplots_kwargs)
        self.fig = self.ax.figure
        self.ax.set(
            xlabel=self.label_for(self.projection[0]),
            ylabel=self.label_for(self.projection[1]),
        )
        self.ax.grid()
        self.ax.axis("equal")

        # create plot elements
        (self.artist_beamline,) = self.ax.plot([], [], "k-")
        self.artist_startpoint = mpl.patches.FancyArrowPatch(
            (0, 0),
            (0, 0),
            mutation_scale=20,
            color="k",
            arrowstyle="-|>",
            zorder=5,
            lw=0,
        )
        self.ax.add_patch(self.artist_startpoint)
        self.artists_boxes = []
        self.artists_labels = []

        # set data
        if survey is not None:
            self.update(survey, line, autoscale=True)

    def update(self, survey, line=None, autoscale=False):
        """
        Update the survey data this plot shows

        Args:
            survey: Survey data.
            line: Line data.
            autoscale: Whether or not to perform autoscaling on all axes
        Returns:
            changed artists
        """

        changed = []

        if self.projection == "3D":
            ...
            raise NotImplementedError()

        else:
            A, B = self.projection
            scale = self.factor_for(A)
            if scale != self.factor_for(B):
                # can't handle this, because angles are not preserved
                raise ValueError(f"Display units for {A} and {B} must be equal!")
            X = get(survey, A) * scale
            Y = get(survey, B) * scale
            # ang: transform angles from data (A-B) to axis (X-Y) coordinate system
            if self.projection == "ZX":
                R = get(survey, "theta")
                ang = lambda a: a
            elif self.projection == "XZ":
                R = get(survey, "theta")
                ang = lambda a: np.pi / 2 - a
            else:
                ...
                raise NotImplementedError()

            NAME = get(survey, "name")
            BEND = get(survey, "angle")

            # beam line
            ############
            self.artist_beamline.set_data(X, Y)
            changed.append(self.artist_beamline)
            self.artist_startpoint.set_positions((2 * X[0] - X[1], 2 * Y[0] - Y[1]), (X[0], Y[0]))
            changed.append(self.artist_startpoint)

            # elements
            ###########
            while len(self.artists_boxes) > 0:
                # remove old artists
                self.artists_boxes.pop().remove()

            while len(self.artists_labels) > 0:
                # remove old artists
                self.artists_labels.pop().remove()

            helicity = 1
            for i, (x, y, rt, name, arc) in enumerate(zip(X, Y, R, NAME, BEND)):

                drift_length = get(survey, "drift_length", None)
                if drift_length is not None and drift_length[i] > 0:
                    continue  # skip drift spaces

                helicity = np.sign(arc) or helicity
                # rt = angle of tangential direction in data coords
                # rr = angle of radial direction (outward) in axis coords
                rr = ang(rt - arc / 2 + helicity * np.pi / 2)

                element = line.element_dict.get(name) if line is not None else None
                order = get(element, "order", None)
                order = get(survey, "order", {i: order})[i]
                length = get(element, "length", None)
                length = get(survey, "length", {i: length})[i]
                if length is not None:
                    length = length * scale

                # box
                ######

                box_style = {}
                if order is not None:
                    box_style["color"] = f"C{order}"
                if length is not None:
                    box_style["length"] = length
                boxes = self.boxes
                if boxes is None:
                    boxes = line is None or order is not None
                box_style = self._get_config(boxes, name, **box_style)

                if box_style is not None:
                    width = box_style.pop("width", self.element_width * scale)
                    length = box_style.pop("length", 0)
                    if length > 0 and arc:
                        rho = length / arc
                        box = mpl.patches.Wedge(
                            **defaults(
                                box_style,
                                center=(
                                    x - helicity * rho * np.cos(rr) / np.cos(arc / 2),
                                    y - helicity * rho * np.sin(rr) / np.cos(arc / 2),
                                ),
                                r=rho + width / 2,
                                width=width,
                                theta1=np.rad2deg(rr - helicity * arc / 2)
                                + 90 * (1 - helicity),  # rr - arc/2),
                                theta2=np.rad2deg(rr + helicity * arc / 2)
                                + 90 * (1 - helicity),  # rr + arc/2),
                                alpha=0.5,
                                zorder=3,
                            )
                        )
                    else:
                        box = mpl.patches.Rectangle(
                            **defaults(
                                box_style,
                                xy=(x - width / 2, y - length / 2),
                                width=width,
                                height=length or (0.1 * scale),
                                angle=np.rad2deg(ang(rt - arc / 2)) - 90,
                                rotation_point="center",
                                alpha=0.5,
                                zorder=3,
                            )
                        )
                    self.ax.add_patch(box)
                    self.artists_boxes.append(box)
                    changed.append(box)

                # label
                ########

                labels = self.labels
                if labels is None:
                    labels = line is not None and order is not None
                label_style = self._get_config(labels, name, text=name)

                if label_style is not None:
                    width = label_style.pop("width", self.element_width * scale)
                    label_style["text"] = label_style["text"].format(name=name)

                    label = self.ax.annotate(
                        **defaults(
                            label_style,
                            xy=(x, y),
                            xytext=(x + 1.5 * width * np.cos(rr), y + 1.5 * width * np.sin(rr)),
                            # xytext=(40*np.cos(rr), 40*np.sin(rr)),
                            # textcoords='offset points',
                            va={1: "bottom", 0: "center", -1: "top"}[np.round(np.sin(rr))],
                            ha={1: "left", 0: "center", -1: "right"}[np.round(np.cos(rr))],
                            # rotation=(np.rad2deg(rr)+90)%180-90,
                            arrowprops=dict(arrowstyle="-", color="0.5", shrinkB=5),
                            clip_on=True,
                            zorder=5,
                        )
                    )
                    self.artists_labels.append(label)
                    changed.append(label)

            if autoscale:
                self.ax.relim()
                datalim = self.ax.dataLim
                self.fig.canvas.draw()  # required to get window extend
                for artist in self.artists_boxes + self.artists_labels:
                    bbox = artist.get_window_extent()
                    datalim = mpl.transforms.Bbox.union(
                        (datalim, bbox.transformed(self.ax.transData.inverted()))
                    )

                self.ax.update_datalim(datalim)
                self.ax.autoscale()

        return changed

    def add_scale(self, scale=None, label=None, *, loc="auto", color="k", fontsize="x-small"):
        """Add a scale patch (a yardstick or ruler)

        Args:
            scale: The length of the scale in data units (typically meter).
            label (str, optional): A label for the scale.
            loc (str): The location of the scale. Can be any of the usual matplotlib locations, e.g. 'auto', 'upper left', 'upper center', 'upper right', 'center left', 'center', 'center right', 'lower left', 'lower center, 'lower right'.
            color: Color for the patch.
            fontsize: Font size of the label.

        Returns:
            The artist added (an AnchoredOffsetbox).
        """
        if scale is None:
            scale = 5
        else:
            # convert to display units
            scale = scale * self.factor_for(self.projection[0])

        if label is None:
            unit = self.display_unit_for(self.projection[0])
            label = f"{scale:g} {unit}"

        super(FloorPlot, self).add_scale(
            self.ax, scale, label=label, loc=loc, color=color, fontsize=fontsize
        )

    @staticmethod
    def _get_config(config, name, **default):
        if isinstance(config, str):
            if re.match(config, name):
                return default
        elif isinstance(config, dict):
            for pattern, args in config.items():
                if re.match(pattern, name):
                    return defaults(args, **default)
        elif config:
            return default


## Restrict star imports to local namespace
__all__ = [
    name
    for name, thing in globals().items()
    if not (name.startswith("_") or isinstance(thing, types.ModuleType))
]
