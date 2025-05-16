"""
Functions for plotting the intermediate results of the Frequent-Yet-Discernible (FYD) Method.
"""

from pathlib import Path

import numpy as np
import igraph as ig
from igraph import Layout


def plot_before_fyd_terms_deletion(
    subgraph,
    subgraph_fuzzy_anchors_vertices_to_delete,
    subgraph_layout,
    visual_style,
    output_dir: Path,
):
    """
    Plots the subgraph before the deletion of the fuzzy linguistic terms.

    Args:
        subgraph: The subgraph.
        subgraph_fuzzy_anchors_vertices_to_delete: The fuzzy anchors vertices to delete.
        subgraph_layout: The subgraph layout.
        visual_style: The visual style.
        output_dir: The path to the output directory.

    Returns:
        The subgraph layout.
    """
    visual_style["vertex_color"] = [
        (
            "#e41a1c"
            if vertex
            in list(
                subgraph_fuzzy_anchors_vertices_to_delete
            )  # mark as soon-to-be deleted
            else visual_style["vertex_color"][idx]
        )  # remain unchanged
        for idx, vertex in enumerate(subgraph.vs)
    ]

    visual_style["vertex_shape"] = [
        "rectangle" if isinstance(vertex["item"], tuple) else "circle"
        for vertex in subgraph.vs
    ]

    ig.plot(
        subgraph,
        target=str(output_dir / "graph_with_marked_vertices_for_deletion.png"),
        layout=subgraph_layout,
        autocurve=True,
        **visual_style,
    )
    return subgraph_layout


def plot_after_fyd_terms_deletion(subgraph, visual_style, output_dir: Path):
    """
    Plots the subgraph after the deletion of the fuzzy linguistic terms.

    Args:
        subgraph:
        visual_style:
        output_dir: The path to the output directory.

    Returns:

    """
    term_vertices, rule_vertices = get_term_and_rule_vertices(subgraph)

    visual_style["vertex_color"] = [
        "#0173b2" if isinstance(vertex["item"], tuple) else "#de8f05"
        for vertex in subgraph.vs
    ]

    # alternative the vertex label angle
    visual_style["vertex_label_angle"] = ([-90] * len(term_vertices)) + [90] * len(
        rule_vertices
    )

    visual_style["vertex_shape"] = [
        "rectangle" if isinstance(vertex["item"], tuple) else "circle"
        for vertex in subgraph.vs
    ]

    # subgraph_layout = subgraph.layout_reingold_tilford_circular()
    # subgraph_layout = subgraph.layout_kamada_kawai()
    subgraph_layout = get_custom_layout(subgraph)
    ig.plot(
        subgraph,
        target=str(output_dir / "graph_with_deletion.png"),
        layout=subgraph_layout,
        autocurve=True,
        **visual_style,
    )
    return subgraph_layout


def plot_before_fyd_rule_deletion(
    subgraph,
    subgraph_layout,
    visual_style,
    vertices_targeted_for_removal,
    output_dir: Path,
):
    """
    Plots the subgraph before the deletion of the rules.

    Args:
        subgraph:
        subgraph_layout:
        visual_style:
        vertices_targeted_for_removal:
        output_dir: The path to the output directory.

    Returns:

    """
    visual_style["vertex_color"] = [
        (
            "#e41a1c"
            if vertex in vertices_targeted_for_removal  # mark as soon-to-be deleted
            else visual_style["vertex_color"][idx]
        )  # remain unchanged
        for idx, vertex in enumerate(subgraph.vs)
    ]

    visual_style["vertex_shape"] = [
        "rectangle" if isinstance(vertex["item"], tuple) else "circle"
        for vertex in subgraph.vs
    ]

    ig.plot(
        subgraph,
        target=str(output_dir / "graph_with_rule_highlight.png"),
        layout=subgraph_layout,
        autocurve=True,
        **visual_style,
    )
    return visual_style


def plot_after_fyd_rule_deletion(subgraph, visual_style, output_dir: Path):
    """
    Plots the subgraph after the deletion of the rules.

    Args:
        subgraph:
        visual_style:
        output_dir: The path to the output directory.

    Returns:

    """
    term_vertices, rule_vertices = get_term_and_rule_vertices(subgraph)
    visual_style["vertex_color"] = [
        "#0173b2" if isinstance(vertex["item"], tuple) else "#de8f05"
        for vertex in subgraph.vs
    ]

    # alternative the vertex label angle
    visual_style["vertex_label_angle"] = ([-90] * len(term_vertices)) + [90] * len(
        rule_vertices
    )

    visual_style["vertex_shape"] = [
        "rectangle" if isinstance(vertex["item"], tuple) else "circle"
        for vertex in subgraph.vs
    ]

    # subgraph_layout = subgraph.layout_reingold_tilford_circular()
    # subgraph_layout = subgraph.layout_kamada_kawai()
    subgraph_layout = get_custom_layout(subgraph)
    ig.plot(
        subgraph,
        target=str(output_dir / "graph_with_rule_deletion.png"),
        layout=subgraph_layout,
        autocurve=True,
        **visual_style,
    )
    return subgraph_layout


def plot_before_fyd(subgraph, output_dir: Path):
    """
    Plots the subgraph before the Frequent-Yet-Discernible algorithm.

    Args:
        subgraph: The subgraph to plot.
        output_dir: The path to the output directory.

    Returns:

    """
    visual_style = {
        "edge_width": 2.0,
        "edge_arrow_width": 0.5,
        "edge_arrow_size": 2.0,
        "edge_curved": 0.1,
        "vertex_size": 20,
        "vertex_color": [
            "#0173b2" if isinstance(vertex["item"], tuple) else "#de8f05"
            for vertex in subgraph.vs
        ],
        "vertex_shape": [
            "rectangle" if isinstance(vertex["item"], tuple) else "circle"
            for vertex in subgraph.vs
        ],
        "vertex_label_size": 30,
        "vertex_label_dist": 2.5,  # distance of label from the vertex
        "vertex_label_color": "#029e73",
        "vertex_label_angle": np.pi / 4,
        # "vertex_label_angle": [0, 90] * (len(subgraph.vs) // 2),
        "margin": 60,  # pads whitespace around the plot
    }

    # add custom labels for the vertices
    term_vertices, rule_vertices = get_term_and_rule_vertices(subgraph)
    term_labels = [
        (
            # f"\u03BC"  # mu
            "    "
            f"{get_unicode_subscript(v['item'][0] + 1)}"  # variable index
            f"\u201A"  # small comma
            f"{get_unicode_subscript(v['item'][1] + 1)}"
        )  # term index
        for v in term_vertices
    ]
    rule_labels = [f"{v['original_ref'] + 1}" for v in rule_vertices]
    subgraph.vs["label"] = term_labels + rule_labels

    # alternative the vertex label angle
    visual_style["vertex_label_angle"] = ([-90] * len(term_vertices)) + [90] * len(
        rule_vertices
    )

    subgraph_layout = get_custom_layout(subgraph)
    # subgraph_layout = subgraph.layout_sugiyama()
    # subgraph_layout = subgraph.layout_kamada_kawai()
    ig.plot(
        subgraph,
        target=str(output_dir / "entire_graph.png"),
        layout=subgraph_layout,
        autocurve=True,
        **visual_style,
    )
    return subgraph_layout, visual_style


def get_term_and_rule_vertices(subgraph):
    """
    Returns the term and rule vertices.

    Args:
        subgraph: The subgraph.

    Returns:
        A tuple consisting of the term and rule vertices.
    """
    term_vertices = [v for v in subgraph.vs if isinstance(v["item"], tuple)]
    # term_vertices = sorted(
    #     [v for v in subgraph.vs if isinstance(v["item"], tuple)],
    #     key=lambda v: (v["item"][0], v["item"][1])
    # )
    rule_vertices = [v for v in subgraph.vs if not isinstance(v["item"], tuple)]
    return term_vertices, rule_vertices


def get_unicode_subscript(number):
    """
    Returns the unicode subscript for a given number.

    Args:
        number: The number.

    Returns:
        The unicode subscript.
    """
    return "".join([chr(8320 + int(digit)) for digit in str(number)])


def get_custom_layout(subgraph):
    term_vertices, rule_vertices = get_term_and_rule_vertices(subgraph)
    term_positions = [
        (idx * 15, 5)
        for idx, v in enumerate(term_vertices)
        if isinstance(v["item"], tuple)
    ]
    try:
        max_term_position = max(pos[0] for pos in term_positions)
    except ValueError:
        max_term_position = 0
    rule_positions = [
        ((idx * (max_term_position / len(rule_vertices))), 0)
        for idx, v in enumerate(rule_vertices)
        if not isinstance(v["item"], tuple)
    ]
    subgraph_layout = Layout(term_positions + rule_positions)
    return subgraph_layout
