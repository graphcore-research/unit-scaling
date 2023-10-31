import webbrowser
from datetime import datetime
from pathlib import Path
from subprocess import Popen

import pandas as pd
import torch
from torch import nn
from typing import *
import matplotlib

from unit_scaling.analysis import example_batch, graph_to_dataframe, prune_non_float_tensors, prune_same_scale_tensors
from unit_scaling.transforms import track_scales, Metrics

from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from torch.fx.graph import Graph
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import colorsys
from torch.fx.node import Node
from math import isnan, isinf
import logging
from copy import copy
import warnings
import pandas as pd
import logging



logger = logging.getLogger(__name__)


def connect_graph_inputs(inter_graph_connections, graph, previous_graphs):
    def find_output_node(input_node):
        graph_idx = len(previous_graphs) - 1
        for prev_graph in reversed(previous_graphs):
            for output_node in reversed(prev_graph.nodes):
                out_users = list(output_node.users.keys())
                if len(output_node.users) == 1 and any(ou.op == "output" for ou in out_users) and input_node.meta["metrics"] == output_node.meta["metrics"]:
                    logger.info("connecting nodes across graph-break: %s %s", input_node.meta["clean_name"], output_node.meta["clean_name"])
                    inter_graph_connections[input_node] = (output_node, graph_idx)
                    input_node.meta["df_drop"] = True
                    return
            graph_idx -= 1
    
    for input_node in graph.nodes:
        if input_node.op == "placeholder":
            find_output_node(input_node)


def tidy_data(
    graphs: List[Graph],
    prune_same_scale: bool = True,
) -> matplotlib.axes.Axes:
    df = pd.DataFrame()
    inter_graph_connections = {}
    pruned_graphs = []
    for graph_idx, graph in enumerate(graphs):
        for nn in graph.nodes:
            print("nn", nn.meta)
        # graph.print_tabular()
        graph = prune_non_float_tensors(graph)
        if prune_same_scale:
            graph = prune_same_scale_tensors(graph)
        connect_graph_inputs(inter_graph_connections, graph, pruned_graphs)
        pruned_graphs.append(graph)
        graph_df = graph_to_dataframe(graph)
        graph_df["graph_idx"] = str(graph_idx)
        df = pd.concat([df, graph_df], ignore_index=True)
    return df


def gen_data(
    model: nn.Module,
    tokenizer: "PreTrainedTokenizerBase",
    batch_size: int,
    seq_len: int,
    backward: bool = True,
    dataset_path: str = "wikitext",
    dataset_name: str = "wikitext-103-v1",
    **plot_kwargs: Any,
) -> matplotlib.axes.Axes:
    inputs, attn_mask, labels = example_batch(
        tokenizer, batch_size, seq_len, dataset_path, dataset_name
    )
    tracked_model = track_scales(model.to("cpu"))
    out = tracked_model(input_ids=inputs, attention_mask=attn_mask, labels=labels)  # TODO: handle
    if backward:
        out.loss.backward()
    graphs = tracked_model.scales_graphs()  # type: ignore[operator]
    return graphs


def write_data(df):
    df = pd.read_csv("data.csv")
    df["Misc"] = df["Misc"] ** -1.0

    dir_path = Path(".loupe_data")
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = Path(".loupe_data") / str(datetime.now())
    df.to_csv(file_path, index=False)
    return file_path


def launch_server(file_path):
    p = Popen(["python", "-m", "http.server"])
    webbrowser.open(
        f"http://localhost:8000/index.html?file_path={file_path}"
    )  # TODO: get any free port
    p.wait()  # TODO: graceful interrupt
    print("goodbye")


def run():
    tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=4*3,
        num_hidden_layers=6,
        num_attention_heads=8,
        initializer_range=1.0,
    )
    model = LlamaForCausalLM(config)
    data = gen_data(
        model,
        tokenizer,
        batch_size=2,
        seq_len=7,
    )
    data = tidy_data(data)
    file_path = write_data(data)
    launch_server(file_path)


if __name__ == "__main__":
    run()
