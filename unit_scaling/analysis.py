# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Tools for analysing scale (and other metrics) within PyTorch models."""

import colorsys
import logging
from math import isnan
from typing import Optional, Tuple

import matplotlib  # type: ignore[import]
import matplotlib.colors  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import pandas as pd
import seaborn as sns  # type: ignore[import]
from torch.fx.graph import Graph
from torch.fx.node import Node

from ._internal_utils import generate__all__
from .transforms import Metrics, prune_non_float_tensors, prune_same_scale_tensors

logger = logging.getLogger(__name__)


def graph_to_dataframe(g: Graph) -> pd.DataFrame:
    """Converts a :class:`torch.fx.Graph` with annotated
    :class:`unit_scaling.transforms.Metrics` into a :class:`pandas.DataFrame`.

    This graph is indended to have been generated by applying
    :func:`unit_scaling.transforms.track_scales` to an arbitrary
    :class:`torch.nn.Module`, running a forward (and possibly backward) pass,
    then calling the `module.scales_graph()` function.

    The resulting dataframe contains all the metrics information for the module,
    and is used internally by the :func:`unit_scaling.analysis.plot` function.

    Args:
        g (Graph): the input graph.

    Returns:
        pd.DataFrame: the metrics dataframe.
    """
    columns = [
        "layer",
        "weight tensor",
        "direction",
        "tensor type",
    ] + Metrics.full_names()
    data = []
    for n in g.nodes:
        # 'output' has to be kept from previous stages to keep fx happy. We drop it here
        if n.name == "output":
            continue
        for direction in ["fwd", "bwd"]:
            tensor_type_prefix = "" if direction == "fwd" else "grad_"
            tensor_type_suffix = "w" if n.meta["requires_grad"] else "x"
            row_data = [
                n.meta["clean_name"],
                n.meta["requires_grad"],
                direction,
                tensor_type_prefix + tensor_type_suffix,
            ]
            for m in Metrics.names():
                directional_metrics = getattr(n.meta["metrics"], direction, None)
                if directional_metrics is not None:
                    v = getattr(directional_metrics, m)
                else:
                    v = None  # pragma: no cover
                row_data.append(v)
            data.append(row_data)

    return pd.DataFrame.from_dict(
        {i: row for i, row in enumerate(data)},
        orient="index",
        columns=columns,
    )


def plot(
    g: Graph,
    title: str = "",
    metric: str = "mean_abs",
    prune_same_scale: bool = True,
    show_arrows: bool = True,
    show_error_bars: bool = True,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
) -> matplotlib.axes.Axes:
    """Generate a :mod:`matplotlib` plot visualising the scales in the forward (and
    optionally backward) pass of all tensors in an arbitrary :class:`torch.nn.Module`.

    The input graph is intended to have been generated by applying
    :func:`unit_scaling.transforms.track_scales` to an arbitrary
    :class:`torch.nn.Module`, running a forward (and possibly backward) pass,
    then calling the `module.scales_graph()` function:

    .. code-block:: python

        from unit_scaling.transforms import track_scales
        from unit_scaling.analysis import plot

        inpt = ...
        model = ...

        model = track_scales(model)
        loss = model(inpt)
        loss.backward()

        graph = model.scales_graph()
        plot(graph)

    Operations that don't output floating-point tensors are automatically pruned from
    the visualised graph, as they are deemed unlikely to be relevant from the
    perspective of model numerics.

    Faint coloured horizontal lines for each row represent error bars indicating
    the maximum and minimum values seen in each tensor during tracking.

    Args:
        g (Graph): the graph to visualise.
        title (str, optional): title for the generated plot. Defaults to "".
        metric (str, optional): the metric to show on the x-axis. Can be any of:
            ("mean_abs", "abs_mean", "std", "abs_max", "abs_min", "numel").
            Defaults to "mean_abs".
        prune_same_scale (bool, optional): prune operations that don't change the scale
            of their input tensors. In practice this means that views / reshapes are not
            shown, making the resulting visualisation clearer. Defaults to True.
        show_arrows (bool, optional): show arrows between operations,
            denoting dependencies. Defaults to True.
        show_error_bars (bool, optional): show max/min error bars. Defaults to True.
        xmin (Optional[float], optional): the minimum x-value to display.
            Defaults to None.
        xmax (Optional[float], optional): the maximum x-value to display.
            Defaults to None.

    Returns:
        matplotlib.axes.Axes: the axes representing the generated plot
    """
    assert metric in Metrics.names() + Metrics.full_names(), (
        f"metric '{metric}' must be one of {Metrics.names()} (these correspond to"
        f" {Metrics.full_names()})"
    )
    full_metric = Metrics.get_full_name(metric)

    g = prune_non_float_tensors(g)
    if prune_same_scale:
        g = prune_same_scale_tensors(g)

    df = graph_to_dataframe(g)

    plot_height = len(df["layer"].unique())
    plt.figure(figsize=(10, plot_height / 4))

    colors = sns.color_palette("colorblind")
    sns.set_palette(colors)

    sns.set_theme()
    p = sns.lineplot(
        data=df,
        x=full_metric,
        y="layer",
        hue="direction",
        hue_order=["fwd", "bwd"],
        style="weight tensor",
        style_order=[False, True],
        dashes=[(0, 1), (0, 1)],
        markers=[".", "v"],
        markersize=9,
        estimator=None,
        orient="y",
    )

    p.set_ylim(plot_height, -1)
    plt.xscale("log", base=2)
    p.xaxis.set_ticks_position("top")
    p.xaxis.set_label_position("top")
    p.xaxis.grid(False)
    p.legend(loc="upper right").set_title("")
    if title:
        p.set_title(title, fontweight="bold")

    plt.axvline(2**-14, color="grey", dashes=(3, 1))
    plt.axvline(2**-7, color="grey", dashes=(1, 3))
    plt.axvline(240, color="grey", dashes=(1, 3))
    plt.axvline(2**16, color="grey", dashes=(3, 1))
    plt.text(
        2**-14,
        plot_height + 0.2,
        "FP16 min,\nFP8 E5 min\n(normal)",
        ha="center",
        va="top",
        size=9,
    )
    plt.text(
        2**-7,
        plot_height + 0.2,
        "FP8 E4 min\n(normal)",
        ha="center",
        va="top",
        size=9,
    )
    plt.text(
        240,
        plot_height + 0.2,
        "FP8 E4 max",
        ha="center",
        va="top",
        size=9,
    )
    plt.text(
        2**16,
        plot_height + 0.2,
        "FP16 max,\nFP8 E5 max",
        ha="center",
        va="top",
        size=9,
    )

    # Cycle through the graph's nodes and give each an index (for the y-axis)
    i = 0
    node_idxs = {}
    for node in g.nodes:
        if node.name != "output":
            name = node.meta["clean_name"]
            if name not in node_idxs:
                node_idxs[name] = i
                i += 1

    min_scale, max_scale = plt.gca().get_xlim()
    if xmin is not None:
        min_scale = xmin
    if xmax is not None:
        max_scale = xmax

    def lighten_color(
        color: Tuple[float, float, float], l_degree: float, s_degree: float
    ) -> Tuple[float, float, float]:
        r, g, b = matplotlib.colors.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        new_l = 1 - l_degree * (1 - l)
        new_s = s_degree * s
        return colorsys.hls_to_rgb(h, new_l, new_s)

    light_colors = [lighten_color(c, l_degree=0.35, s_degree=0.45) for c in colors]

    def draw_error_bar(node: Node, direction: str) -> None:
        metrics = node.meta["metrics"]
        if direction == "bwd" and metrics.bwd is None:  # pragma: no cover
            return

        directional_metrics = getattr(metrics, direction)
        x1, x2 = directional_metrics.abs_min, directional_metrics.abs_max
        y = node_idxs[node.meta["clean_name"]] + (-0.1 if direction == "fwd" else 0.1)
        color = light_colors[0 if direction == "fwd" else 1]
        plt.plot(
            [x1, x2],
            [y, y],
            color=color,
            linestyle="-",
            linewidth=1,
            marker="",
            zorder=1,
        )
        for x in [x1, x2]:
            plt.plot(
                [x, x],
                [y - 0.2, y + 0.2],
                color=color,
                linestyle="-",
                linewidth=1,
                marker="",
                zorder=1,
            )
        plt.gca().set_xlim(min_scale, max_scale)

    def draw_arrow(node_a: Node, node_b: Node, direction: str) -> None:
        a_metrics = node_a.meta["metrics"]
        b_metrics = node_b.meta["metrics"]
        if direction == "bwd" and (  # pragma: no cover
            a_metrics.bwd is None or b_metrics.bwd is None
        ):
            return  # pragma: no cover

        a_x = getattr(getattr(a_metrics, direction), metric)
        b_x = getattr(getattr(b_metrics, direction), metric)
        a_y = node_idxs[node_a.meta["clean_name"]]
        b_y = node_idxs[node_b.meta["clean_name"]]

        annotation = ""
        if a_x == 0 or isnan(a_x):  # pragma: no cover
            a_x = min_scale
        if isnan(a_x):  # pragma: no cover
            logging.warning(f"Node '{node_a.meta['clean_name']}' is NaN. Plotting as 0")
            a_x = min_scale
        if b_x == 0:  # pragma: no cover
            b_x = min_scale
            annotation = "0"
        if isnan(b_x):  # pragma: no cover
            logging.warning(f"Node '{node_b.meta['clean_name']}' is NaN. Plotting as 0")
            b_x = min_scale
            annotation = "0"

        if direction == "fwd":
            color = colors[0]
        else:
            assert direction == "bwd", direction
            color = colors[1]
            a_x, a_y, b_x, b_y = b_x, b_y, a_x, a_y

        plt.annotate(
            annotation,
            color=color,
            va="center",
            xy=((a_x, a_y)),
            xytext=((b_x, b_y)),
            arrowprops=dict(arrowstyle="->", color=color),
        )

    if show_arrows:
        for n in g.nodes:
            if n.name != "output":
                for direction in ["fwd", "bwd"]:
                    for arg in n.args:
                        if isinstance(arg, Node):
                            draw_arrow(n, arg, direction)

    if show_error_bars:
        for n in g.nodes:
            if n.name != "output":
                for direction in ["fwd", "bwd"]:
                    draw_error_bar(n, direction)

    return p


__all__ = generate__all__(__name__)
