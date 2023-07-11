import colorsys
import logging
from math import isnan
from typing import Optional

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.fx.graph import Graph
from torch.fx.node import Node

logger = logging.getLogger(__name__)

from .transforms._track_scales import (
    Metrics,
    prune_non_float_tensors,
    prune_same_scale_tensors,
)


def graph_to_dataframe(g: Graph) -> pd.DataFrame:
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
                    v = None
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
    plt.figure(figsize=(10, plot_height // 4))

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

    def lighten_color(color, l_degree, s_degree):
        r, g, b = matplotlib.colors.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        new_l = 1 - l_degree * (1 - l)
        new_s = s_degree * s
        return colorsys.hls_to_rgb(h, new_l, new_s)

    light_colors = [lighten_color(c, l_degree=0.35, s_degree=0.45) for c in colors]

    def draw_error_bar(node: Node, direction: str) -> None:
        metrics = node.meta["metrics"]
        if direction == "bwd" and metrics.bwd is None:
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
        if direction == "bwd" and (a_metrics.bwd is None or b_metrics.bwd is None):
            return

        a_x = getattr(getattr(a_metrics, direction), metric)
        b_x = getattr(getattr(b_metrics, direction), metric)
        a_y = node_idxs[node_a.meta["clean_name"]]
        b_y = node_idxs[node_b.meta["clean_name"]]

        annotation = ""
        if a_x == 0 or isnan(a_x):
            a_x = min_scale
        if isnan(a_x):
            logging.warning(f"Node '{node_a.meta['clean_name']}' is NaN. Plotting as 0")
            a_x = min_scale
        if b_x == 0:
            b_x = min_scale
            annotation = "0"
        if isnan(b_x):
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
