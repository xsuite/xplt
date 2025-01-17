#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting lines

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-08"

import re
import numpy as np
import matplotlib as mpl
from .util import (
    get,
    iter_elements,
    defaults,
    defaults_for,
    PUBLIC_SECTION_END,
    PUBLIC_SECTION_BEGIN,
)
from .base import XPlot, XManifoldPlot
from .properties import Property, DataProperty


def element_strength(element, n):
    """Get knl strength of element"""
    if hasattr(element, f"k{n}") and hasattr(element, "length"):
        return getattr(element, f"k{n}") * element.length
    elif hasattr(element, "knl") and n <= element.order:
        return element.knl[n]
    return 0


def nominal_order(element):
    """Get nominal element order (even if coefficients might be zero)"""
    if element.__class__.__name__ == "Bend":  # avoid using type to support Views
        return 0
    if hasattr(element, "length"):
        for n in range(10, -1, -1):
            if hasattr(element, f"k{n}") or hasattr(element, f"k{n}s"):
                return n
    if hasattr(element, "order"):
        return int(element.order)
    if hasattr(element, "knl"):
        return len(element.knl)
    return -1


def effective_order(element):
    """Get effective order of element (ignoring zero strength)"""
    order = {-1}
    if hasattr(element, "length") and element.length > 0:
        for n in range(10, -1, -1):
            if get(element, f"k{n}", 0) or get(element, f"k{n}s", 0):
                order.add(n)
    for knl in ("knl", "ksl"):
        if hasattr(element, knl):
            order.add(int(np.max(np.nonzero(getattr(element, knl)), initial=-1)))
    return max(order)


def order(knl):
    """Get order of knl string as int"""
    return int(re.match(r"k(\d+)l", knl).group(1))


def tanc(x):
    """Tangens cardinalis, i.e. tan(x)/x with limit tanc(0)=1"""
    # Note that np.sinc is sin(pi*x)/(pi*x) and not sin(x)/x !
    return np.sinc(x / np.pi) / np.cos(x)


PUBLIC_SECTION_BEGIN()


class KnlPlot(XManifoldPlot):
    """A plot for knl values along line"""

    def __init__(self, line=None, *, knl=None, filled=True, resolution="auto", **kwargs):
        """

        Args:
            line (xtrack.Line): Line of elements.
            knl (int | list[int] | str): Maximum order or list of orders n to plot knl values for.
                This can also be a manifold subplot specification string like ``"k0l+k1l,k2l"``,
                see :class:`~.base.XManifoldPlot` for details.
                If None, automatically determine from line.
            filled (bool): If True, make a filled plot instead of a line plot.
            resolution (int | string): Number of points to use for plotting. Use "auto" for
                a point at the start/end of each element.
            kwargs: See :class:`~.base.XPlot` for additional arguments

        Known issues:
            - Thin elements produced with MAD-X MAKETHIN do overlap due to the displacement introduced by the TEAPOT algorithm.
              This leads to glitches of knl being doubled or zero at element overlaps for lines containing such elements.

        """

        if knl is None:
            if line is None:
                raise ValueError("Either line or knl parameter must not be None")
            knl = int(max([effective_order(e) for e in line.elements]))
        if isinstance(knl, int):
            knl = range(knl + 1)
        if not isinstance(knl, str):
            knl = [[[f"k{n}l" for n in knl]]]
        self.resolution = resolution
        self.filled = filled

        super().__init__(on_x="s", on_y=knl, **kwargs)

        # create plot elements
        def create_artists(i, j, k, a, p):
            kwargs = dict(color=f"C{order(p)}", alpha=0.5, label=self.label_for(p, unit=True))
            if self.filled:
                kwargs.update(zorder=3, lw=0)
                artist = a.fill_between([], [], **kwargs)
                artist._constructor_kwargs = kwargs
                return artist
            else:
                return a.plot([], [], **kwargs)[0]

        self._create_artists(create_artists)

        for a in self.axflat:
            a.axhline(0, c="k", lw=1, zorder=4)
        self.legend(show="auto", ncol=5)

        # set data
        if line is not None:
            self.update(line)

    def update(self, line, *, autoscale=None):
        """
        Update the line data this plot shows

        Args:
            line (xtrack.Line): Line of elements.
            autoscale (str | None | bool): Whether and on which axes to perform autoscaling.
                One of `"x"`, `"y"`, `"xy"`, `False` or `None`. If `None`, decide based on :meth:`matplotlib.axes.Axes.get_autoscalex_on` and :meth:`matplotlib.axes.Axes.get_autoscaley_on`.


        Returns:
            changed artists
        """
        # compute knl as function of s
        if self.resolution == "auto":
            S = set()
            for name, el, s0, s1 in iter_elements(line):
                S.update({s0, s1})
            S = np.array(sorted(list(S)))
        else:
            S = np.linspace(0, line.get_length(), self.resolution)
        values = {p: np.zeros_like(S) for p in self.on_y_unique}
        for name, el, s0, s1, mask in iter_elements(line, s=S):
            for knl in self.on_y_unique:
                n = order(knl)
                values[knl][mask] += element_strength(el, n)
        if self.resolution == "auto":
            S = np.repeat(S, 2)[1:]
            values = {p: np.repeat(v, 2)[:-1] for p, v in values.items()}

        # plot
        s = S * self.factor_for("s")
        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                ax = self.axis(i, j)
                for k, p in enumerate(pp):
                    art = self.artists[i][j][k]
                    y = self.factor_for(p) * values[p]
                    if self.filled:
                        kwargs = art._constructor_kwargs
                        changed.append(art)
                        art.remove()
                        self.artists[i][j][k] = art = ax.fill_between(S, 0, y, **kwargs)
                        art._constructor_kwargs = kwargs
                    else:
                        art.set_data((s, y))
                    changed.append(art)

                # autoscale
                self._autoscale(ax, autoscale, tight="x")

        return changed

    def prop(self, p):
        if match := re.fullmatch(r"k(\d+)l", p):
            n = match.group(1)
            return Property(symbol=f"$k_{n}l$", unit="rad" if n == "0" else f"m^-{n}")
        return super().prop(p)

    def label_for(self, *pp, unit=True, description=True):
        """
        Return axis label for list of properties, joining where possible

        Args:
            pp: Property names
            unit (bool): Whether to include unit
            description (bool): Whether to include description

        Returns:
            str: Axis label
        """
        if len(pp) > 1 and np.all([re.match(r"k\d+l", p) for p in pp]):
            label = "$k_nl$"
            if unit:
                label += " / $m^{-n}$"
            return label
        return super().label_for(*pp, unit=unit, description=description)


class FloorPlot(XPlot):
    """A floor plan of the line based on survey data"""

    def __init__(
        self,
        survey=None,
        line=None,
        projection="ZX",
        *,
        default_boxes=None,
        boxes=None,
        labels=False,
        ignore=None,
        element_width=1,
        axis="equal",
        **kwargs,
    ):
        """

        Args:
            survey (Any | None): Survey data from MAD-X or Xsuite.
            line (xtrack.Line | None): Line data with additional information about elements.
                Use this to have colored boxes of correct size etc.
            projection (str): The projection to use: A pair of coordinates ('XZ', 'ZY' etc.)
            boxes (None | bool | str | iterable | dict): Config option for showing colored boxes for elements. See below.
                Detailed options can be "length" and all options suitable for a patch, such as "color", "alpha", etc.
                By default, all elements with a multipole order are shown, or if line is None then all element.
            default_boxes (bool): Whether to keep the default boxes even if custom box options are specified.
            labels (None | bool | str | iterable | dict): Config option for showing labels for elements. See below.
                Detailed options can be "text" (e.g. "Dipole {name}" where name will be
                replaced with the element name) and all options suitable for an annotation,
                such as "color", "alpha", etc.
            ignore (None | str | list[str]): Optional patter or list of patterns to ignore elements with matching names.
                Note that drift spaces are always ignored.
            element_width (float): Width of element boxes.
            axis (str): Aspect ratio of the plot. Default is 'equal'.
            kwargs: See :class:`~.base.XPlot` for additional arguments


        The config options passed to boxes and labels can be:
            - None: Use good defaults.
            - A bool: En-/disable option for all elements (except drifts).
            - A str (regex): Filter by element name.
            - A list, tuple or numpy array: Filter by any of the given element names
            - A dict: Detailed options to apply for each element in the form of
              `{"regex": {...}}`. For each matching element name, the options are used.

        """

        kwargs["_properties"] = defaults(
            kwargs.get("_properties"),
            X=DataProperty("X", "m"),
            Y=DataProperty("Y", "m"),
            Z=DataProperty("Z", "m"),
            theta=DataProperty("theta", "rad", "$\\Theta$"),
            phi=DataProperty("phi", "rad", "$\\Phi$"),
            psi=DataProperty("psi", "rad", "$\\Psi$"),
        )

        super().__init__(**kwargs)

        if projection == "3D":
            raise NotImplementedError()

        self.projection = projection
        self.boxes = boxes
        self.default_boxes = default_boxes if default_boxes is not None else (boxes is not False)
        self.labels = labels
        self.ignore = [ignore] if isinstance(ignore, str) else ignore
        self.element_width = element_width

        if isinstance(self.boxes, (list, tuple, np.ndarray)):
            self.boxes = "|".join(["^" + ll + "$" for ll in self.boxes])
        if isinstance(self.labels, (list, tuple, np.ndarray)):
            self.labels = "|".join(["^" + ll + "$" for ll in self.labels])

        # Create plot
        self.ax.set(
            xlabel=self.label_for(self.projection[0]), ylabel=self.label_for(self.projection[1])
        )
        self.ax.axis(axis)

        # create plot elements
        (self.artist_beamline,) = self.ax.plot([], [], "k-")
        self.artist_startpoint = mpl.patches.FancyArrowPatch(
            (0, 0), (0, 0), mutation_scale=20, color="k", arrowstyle="-|>", zorder=5, lw=0
        )
        self.ax.add_patch(self.artist_startpoint)
        self.artists_boxes = []
        self.artists_labels = []

        # set data
        if survey is not None or line is not None:
            self.update(survey, line)

    def update(self, survey=None, line=None, *, autoscale=None):
        """
        Update the survey data this plot shows

        Args:
            survey (Any | None): Survey data. Defaults to `line.survey()`.
                For convenience, passing line as first argument is also supported.
            line (None | xtrack.Line): Line object. Defaults to `survey.line` if possible.
            autoscale (str | None | bool): Whether and on which axes to perform autoscaling.
                One of `"x"`, `"y"`, `"xy"`, `False` or `None`. If `None`, decide based on :meth:`matplotlib.axes.Axes.get_autoscalex_on` and :meth:`matplotlib.axes.Axes.get_autoscaley_on`.

        Returns:
            changed artists

        """

        # Handle various input types in a smart way
        if line is None and type(survey).__name__ == "Line":
            survey, line = None, survey
        if survey is None and line is not None:
            survey = line.survey()
        if line is None and survey is not None:
            line = getattr(survey, "line", None)

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

                def ang(a):
                    return a

            elif self.projection == "XZ":
                R = get(survey, "theta")

                def ang(a):
                    return np.pi / 2 - a

            elif self.projection == "ZY":
                R = get(survey, "theta")

                def ang(a):
                    return a

            else:
                ...
                raise NotImplementedError()

            NAME = get(survey, "name")
            BEND = get(survey, "angle")

            # beam line
            ############
            self.artist_beamline.set_data(X, Y)
            changed.append(self.artist_beamline)
            # start point arrow
            i = np.argmax((X > X[0]) | (Y > Y[0]))
            self.artist_startpoint.set_positions((2 * X[0] - X[i], 2 * Y[0] - Y[i]), (X[0], Y[0]))
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
            legend_entries = []
            for i, (x, y, rt, name, arc) in enumerate(zip(X, Y, R, NAME, BEND)):
                helicity = np.sign(arc) or helicity
                # rt = angle of tangential direction in data coords
                # rr = angle of radial direction (outward) in axis coords
                rr = ang(rt - arc / 2 + helicity * np.pi / 2)

                drift_length = get(get(survey, "drift_length", []), i, -1)
                order = get(get(survey, "order", []), i, -1)
                length = get(get(survey, "length", []), i, 0)
                is_thick = False
                element = None

                try:
                    element = line[name]
                    is_thick = element.isthick
                    if type(element).__name__ == "Replica":
                        name = element.resolve(line, get_name=True)
                    if order < 0:
                        order = nominal_order(element)
                    if not length:
                        length = get(element, "length", None)
                except (TypeError, KeyError):
                    pass

                # ignored elements
                if drift_length > 0 and not is_thick or type(element).__name__ == "Drift":
                    continue  # always skip drift spaces
                if self.ignore is not None:
                    if np.any([re.match(pattern, name) is not None for pattern in self.ignore]):
                        continue  # skip ignored
                if name == "_end_point":
                    continue

                # box
                ######

                # default style
                default_box_style = dict(
                    color=f"C{order}" if order >= 0 else "k",
                    length=length or 0,
                    label={
                        0: "Bending magnet" if arc else None,
                        1: "Quadrupole magnet",
                        2: "Sextupole magnet",
                        3: "Octupole magnet",
                    }.get(order),
                )

                boxes = self.boxes
                if boxes is None:
                    boxes = line is None or order >= 0
                box_style = self._get_config(boxes, name, **default_box_style)
                if box_style is None and self.default_boxes and (line is None or order >= 0):
                    box_style = default_box_style

                if box_style is not None:
                    width = box_style.pop("width", self.element_width) * scale
                    length = box_style.pop("length", 0) * scale
                    if box_style.get("label") in legend_entries:
                        box_style.pop("label")  # prevent duplicate legend entries
                    else:
                        legend_entries.append(box_style.get("label"))

                    # Handle thick elements
                    if is_thick:
                        # Find the center of single kick for equivalent thin element
                        d = length * tanc(arc / 2) / 2
                        x += d * np.cos(ang(rt))
                        y += d * np.sin(ang(rt))

                    if length > 0 and arc:
                        # bending elements as wedge
                        rho = length / arc
                        x_center = x - helicity * rho * np.cos(rr) / np.cos(arc / 2)
                        y_center = y - helicity * rho * np.sin(rr) / np.cos(arc / 2)
                        wedge_kwargs = defaults_for(
                            mpl.patches.Wedge,
                            box_style,
                            center=(x_center, y_center),
                            r=rho + width / 2,
                            width=width,
                            theta1=np.rad2deg(rr - helicity * arc / 2)
                            + 90 * (1 - helicity),  # rr - arc/2),
                            theta2=np.rad2deg(rr + helicity * arc / 2)
                            + 90 * (1 - helicity),  # rr + arc/2),
                            alpha=0.5,
                            zorder=3,
                        )
                        box = mpl.patches.Wedge(**wedge_kwargs)
                    else:
                        # other elements as rect
                        box = mpl.patches.Rectangle(
                            **defaults_for(
                                mpl.patches.Rectangle,
                                box_style,
                                xy=(x - width / 2, y - length / 2),
                                width=width,
                                height=length,
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
                    labels = line is not None and order >= 0
                label_style = self._get_config(labels, name, text=name)

                if label_style is not None:
                    width = label_style.pop("width", self.element_width * scale)
                    label_style["text"] = label_style["text"].format(name=name, element=element)

                    label = self.ax.annotate(
                        **defaults_for(
                            "text",
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

            # autoscale
            self._autoscale(
                self.ax,
                autoscale,
                artists=self.artists_boxes + self.artists_labels + [self.artist_beamline],
            )

        return changed

    def legend(self, **kwargs):
        self.ax.legend(**kwargs)

    def add_scale(self, scale=None, label=None, *, loc="auto", color="k", fontsize="x-small"):
        """Add a scale patch (a yardstick or ruler)

        Args:
            scale (float): The length of the scale in data units (typically meter).
            label (str | None): A label for the scale.
            loc (str): The location of the scale. Can be any of the usual matplotlib locations,
                e.g. 'auto', 'upper left', 'upper center', 'upper right', 'center left', 'center',
                'center right', 'lower left', 'lower center, 'lower right'.
            color (Any): Color for the patch.
            fontsize (Any): Font size of the label.

        Returns:
            matplotlib.offsetbox.AnchoredOffsetbox: The artist added
        """
        if scale is None:
            scale = 5
        else:
            # convert to display units
            scale = scale * self.factor_for(self.projection[0])

        if label is None:
            unit = self.display_unit_for(self.projection[0])
            label = f"{scale:g} {unit}"

        return super().add_scale(
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
                    if args is False:
                        return None
                    if isinstance(args, dict):
                        return defaults(args, **default)
                    return default
        elif config:
            return default


__all__ = PUBLIC_SECTION_END()
