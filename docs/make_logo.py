"""Generate the quadax logo, wordmark and favicon.

Run from the ``docs`` directory to regenerate everything in ``_static/images``::

    python make_logo.py

The mark is the trapezoidal rule applied to a cubic: panels approximate the area
under the curve, and the gaps between the chords and the curve they cut across
are the quadrature error. Only the matplotlib-bundled DejaVu fonts are used, so
the output is reproducible on any machine.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

# -- palette ------------------------------------------------------------------

INK = "#2B0B3F"  # tile background
PAPER = "#FFF2E0"  # curve, axis and nodes
WORDMARK_BALANCED = "#9B3FBF"
WORDMARK_LIGHT = PAPER

PANEL_COLORS = ["#5C0F87", "#8A23A6", "#C33F94", "#F2704F", "#FFC85E"]
PANEL_CMAP = LinearSegmentedColormap.from_list("quadax", PANEL_COLORS)

# -- the integrand ------------------------------------------------------------

# Turning points of the cubic: it rises to a maximum at TURN_UP, falls to a
# minimum at TURN_DOWN and rises again.
TURN_UP, TURN_DOWN = 0.30, 0.78
BASE, TOP = 0.09, 0.95  # the range the curve is scaled into, in plot heights

N_PANELS = 5
N_PANELS_COARSE = 3  # for sizes too small to resolve the full set


def _cubic(x):
    return x**3 / 3 - (TURN_UP + TURN_DOWN) * x**2 / 2 + TURN_UP * TURN_DOWN * x


_SPAN = _cubic(np.linspace(0.0, 1.0, 4001))


def integrand(x):
    """The cubic the mark integrates, scaled to run between BASE and TOP."""
    raw = _cubic(np.asarray(x, dtype=float))
    return BASE + (TOP - BASE) * (raw - _SPAN.min()) / (_SPAN.max() - _SPAN.min())


# -- drawing ------------------------------------------------------------------

# Geometry of the mark, in units of the tile's side length. The tile occupies
# [0, 1] x [0, 1]; the plot sits inside it with room for the tick marks below.
PLOT_X0, PLOT_WIDTH = 0.135, 0.73
PLOT_Y0, PLOT_HEIGHT = 0.225, 0.645
AXIS_OVERHANG = 0.05
TICK_LENGTH = 0.048

# Stroke weights, also as fractions of the tile side, so that every rendering
# size gets the same drawing rather than the same absolute line widths.
CURVE_WEIGHT = 0.030
GAP_WEIGHT = 0.017
TICK_WEIGHT = 0.021
NODE_SIZE = 0.052
CORNER_RADIUS = 0.225


def draw_mark(ax, unit, x=0.0, y=0.0, coarse=False, ticks=True):
    """Draw the square mark with its lower left corner at ``(x, y)``.

    ``unit`` is the tile's side length measured in points, which converts the
    fractional stroke weights above into matplotlib line widths.
    """
    npanels = N_PANELS_COARSE if coarse else N_PANELS
    edges = np.linspace(0.0, 1.0, npanels + 1)

    tile = FancyBboxPatch(
        (x, y),
        1.0,
        1.0,
        boxstyle=f"round,pad=0,rounding_size={CORNER_RADIUS}",
        facecolor=INK,
        edgecolor="none",
        zorder=1,
    )
    ax.add_patch(tile)

    def to_x(t):
        return x + PLOT_X0 + PLOT_WIDTH * np.asarray(t, dtype=float)

    def to_y(v):
        return y + PLOT_Y0 + PLOT_HEIGHT * np.asarray(v, dtype=float)

    heights = integrand(edges)
    for i in range(npanels):
        # Each panel is the trapezoid under the chord joining the curve at the
        # panel's two edges, so it visibly falls short of the curve where the
        # curve is concave and overshoots it where the curve is convex.
        ax.fill_between(
            to_x(edges[i : i + 2]),
            to_y(0),
            to_y(heights[i : i + 2]),
            facecolor=PANEL_CMAP(i / (npanels - 1)),
            # The gap between panels is drawn as an edge in the tile colour.
            edgecolor=INK,
            linewidth=GAP_WEIGHT * unit,
            zorder=2,
        )

    if ticks:
        for edge in edges:
            ax.plot(
                to_x([edge, edge]),
                [to_y(0) - TICK_LENGTH, to_y(0)],
                color=PAPER,
                linewidth=TICK_WEIGHT * unit,
                solid_capstyle="butt",
                zorder=3,
            )

    t = np.linspace(0.0, 1.0, 2000)
    ax.plot(
        to_x(t),
        to_y(integrand(t)),
        color=PAPER,
        linewidth=CURVE_WEIGHT * unit,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        to_x([-AXIS_OVERHANG, 1 + AXIS_OVERHANG]),
        to_y([0, 0]),
        color=PAPER,
        linewidth=CURVE_WEIGHT * unit,
        solid_capstyle="round",
        zorder=4,
    )
    # The nodes are where the rule samples the integrand, and pin the corners of
    # the trapezoids to the curve.
    ax.plot(
        to_x(edges),
        to_y(heights),
        "o",
        color=PAPER,
        markersize=NODE_SIZE * unit,
        zorder=5,
    )


# Wordmark proportions, again relative to the tile side length.
WORD_GAP = 0.18
WORD_XHEIGHT = 0.30

_FONT = FontProperties(family="DejaVu Sans Mono", weight="bold")


def draw_wordmark(ax, color, x=0.0, y=0.0):
    """Draw "quadax" as outlines, x-height band centred on ``y``.

    The glyphs are emitted as paths rather than text so that the SVG carries no
    font dependency. Returns the advance width, in tile side lengths.
    """
    glyphs = TextPath((0, 0), "quadax", size=1.0, prop=_FONT)
    scale = WORD_XHEIGHT / TextPath((0, 0), "x", size=1.0, prop=_FONT).get_extents().y1
    ink = glyphs.get_extents()
    transform = Affine2D().scale(scale)
    transform.translate(x - ink.x0 * scale, y - WORD_XHEIGHT / 2)
    outlines = transform.transform_path(glyphs)
    ax.add_patch(PathPatch(outlines, facecolor=color, edgecolor="none", zorder=5))
    return ink.width * scale


def _figure(width, height, pixels):
    """A figure of ``width`` x ``height`` tile units, ``pixels`` tall, no margins."""
    dpi = 100.0
    fig = plt.figure(figsize=(width * pixels / height / dpi, pixels / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    return fig, ax, pixels * 72.0 / dpi / height


def make_lockup(pixels=512, color=WORDMARK_BALANCED):
    """Mark and wordmark side by side, on a transparent background."""
    width = 1.0 + WORD_GAP + 1.9  # trimmed to the true text width below
    fig, ax, unit = _figure(width, 1.0, pixels)
    draw_mark(ax, unit)
    advance = draw_wordmark(ax, color, x=1.0 + WORD_GAP, y=0.5)
    width = 1.0 + WORD_GAP + advance
    ax.set_xlim(0, width)
    fig.set_size_inches(fig.get_figheight() * width, fig.get_figheight())
    return fig


def make_mark(pixels=512, coarse=False, ticks=True):
    """The square mark on its own, on a transparent background."""
    fig, ax, unit = _figure(1.0, 1.0, pixels)
    draw_mark(ax, unit, coarse=coarse, ticks=ticks)
    return fig


def main(outdir="_static/images"):
    """Write every logo file the README and the docs refer to."""
    import os

    os.makedirs(outdir, exist_ok=True)

    def save(fig, name, **kwargs):
        for ext in ("svg", "png"):
            fig.savefig(
                os.path.join(outdir, f"{name}.{ext}"), transparent=True, **kwargs
            )
        plt.close(fig)

    save(make_lockup(color=WORDMARK_BALANCED), "logo")
    save(make_lockup(color=WORDMARK_LIGHT), "logo_light")
    save(make_mark(), "logo_mark")

    # The docs sidebar renders the lockup a couple of hundred pixels wide.
    fig = make_lockup(pixels=160, color=WORDMARK_LIGHT)
    fig.savefig(os.path.join(outdir, "logo_small.png"), transparent=True)
    plt.close(fig)

    # Favicon sizes are too small for the full set of panels or the tick marks.
    from PIL import Image

    frames = []
    for size in (64, 48, 32, 16):
        path = os.path.join(outdir, f"_favicon_{size}.png")
        fig = make_mark(pixels=size, coarse=size <= 32, ticks=size > 32)
        fig.savefig(path, transparent=True)
        plt.close(fig)
        frames.append(Image.open(path).convert("RGBA"))
        os.remove(path)
    # Pillow drops any requested size larger than the base image, so the base
    # has to be the largest frame.
    frames[0].save(
        os.path.join(outdir, "favicon.ico"),
        sizes=[f.size for f in frames],
        append_images=frames[1:],
    )


if __name__ == "__main__":
    main()
