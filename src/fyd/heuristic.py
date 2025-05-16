"""
Implements the Frequent-Yet-Discernible Method. More specifically, this script contains the
primary function, as well as its supporting helper methods that assist in calculating the heuristic.
"""

import pathlib
from typing import List, Tuple, Union, Set

import torch
import igraph
import numpy as np
from kneed import KneeLocator  # "Kneedle" algorithm, https://github.com/arvkevi/kneed
from fuzzy.sets import FuzzySetGroup

# TODO: fix the plots of the FYD subgraph
# from fyd.visualize import (
#     plot_before_fyd,
#     plot_before_fyd_terms_deletion,
#     plot_after_fyd_terms_deletion,
#     plot_before_fyd_rule_deletion,
#     plot_after_fyd_rule_deletion,
# )


def frequent_discernible(
    input_data,
    knowledge_base: "KnowledgeBase",
    device: torch.device,
    output_dir: Union[None, pathlib.Path] = None,
    verbose: bool = False,
):
    """
    The fuzzy logic rule generation method called the Frequent But Discernible Method.

    Args:
        input_data:
        knowledge_base:
        device: The device to use.
        output_dir: The path to the output directory. If None, no plots will be saved.
        verbose: Whether to print debug information.

    Returns:

    """
    # make the directory if it does not exist
    if output_dir is None:
        output_dir = pathlib.Path(__file__).parent.parent.parent  # overwrite for demo purposes
        output_dir = output_dir / "plots"
        output_dir.mkdir(exist_ok=True, parents=True)

    fuzzy_anchors_vertices: igraph.VertexSeq = knowledge_base.select_by_tags(
        tags={"premise", "anchor"}
    )
    subgraph = induce_subgraph(fuzzy_anchors_vertices, knowledge_base)

    # if output_dir is not None:
    #     subgraph_layout, visual_style = plot_before_fyd(subgraph, output_dir=output_dir)

    # find the fuzzy_anchors_vertices in the new subgraph
    subgraph_fuzzy_anchors_vertices: igraph.VertexSeq = subgraph.vs.select(
        tags_eq={"premise", "anchor"}
    )

    print(f"There are {len(fuzzy_anchors_vertices)} terms across all rules.")
    vertices_heuristics = calc_fyd_heuristic(
        fuzzy_anchors_vertices,
        input_data.data,
        knowledge_base,
        subgraph,
        subgraph_fuzzy_anchors_vertices,
        device=device,
    )
    # find the bend of the plot & determine vertices to be removed, but their ids are subgraph
    cutoff_value = find_heuristic_cutoff(vertices_heuristics)
    if np.isclose(cutoff_value, 0.0):
        subgraph_vertices_to_delete = np.where(vertices_heuristics <= cutoff_value)[0]
    else:
        subgraph_vertices_to_delete = np.where(vertices_heuristics < cutoff_value)[0]
    subgraph_fuzzy_anchors_vertices_to_delete = [
        subgraph_fuzzy_anchors_vertices[idx] for idx in subgraph_vertices_to_delete
    ]
    if len(subgraph_fuzzy_anchors_vertices_to_delete) == len(
        subgraph_fuzzy_anchors_vertices
    ):
        subgraph_vertices_to_delete = np.where(
            vertices_heuristics <= np.median(vertices_heuristics)
        )[0]
        subgraph_fuzzy_anchors_vertices_to_delete = [
            subgraph_fuzzy_anchors_vertices[idx] for idx in subgraph_vertices_to_delete
        ]
    vertices_items_to_delete = [
        v["item"] for v in subgraph_fuzzy_anchors_vertices_to_delete
    ]
    vertices_to_delete = knowledge_base.graph.vs.select(
        item_in=vertices_items_to_delete
    )

    # if output_dir is not None:
    #     plot_before_fyd_terms_deletion(
    #         subgraph,
    #         subgraph_fuzzy_anchors_vertices_to_delete,
    #         subgraph_layout,
    #         visual_style,
    #         output_dir=output_dir,
    #     )
    subgraph.delete_vertices(subgraph_vertices_to_delete.tolist())
    # if output_dir is not None:
    #     subgraph_layout = plot_after_fyd_terms_deletion(
    #         subgraph, visual_style, output_dir=output_dir
    #     )

    if verbose:
        print(
            f"Deleting {len(vertices_to_delete)} vertices from the KnowledgeBase graph."
        )

    knowledge_base.graph.delete_vertices(vertices_to_delete)

    # delete unconnected vertices (some rule vertices can become unconnected)
    # knowledge_base.graph.vs.select(_degree=0).delete()

    # np.where((1 - ((np.array(
    #     subgraph.degree(subgraph_fuzzy_anchors_vertices)) / len(subgraph_rule_vertices)) *
    #                subgraph.closeness(subgraph_fuzzy_anchors_vertices))) > avg)[0].shape
    # some helpful graph theory functions
    # subgraph.similarity_dice(subgraph_fuzzy_anchors_vertices)
    # subgraph.harmonic_centrality(subgraph_fuzzy_anchors_vertices)
    # import numpy as np
    # import matplotlib.pyplot as plt
    # plt.plot(range(446), sorted([
    #     v for v in subgraph.closeness(subgraph_fuzzy_anchors_vertices) if not np.isnan(v)]))

    rules = knowledge_base.rules
    rule_vertices_to_delete = []
    for rule in rules:
        delete_rule = False
        premise: Set[Tuple[int, int]] = set(rule.premise.indices[0])
        for other_rule in rules:
            other_premise: Set[Tuple[int, int]] = set(other_rule.premise.indices[0])
            if premise != other_premise and premise.issubset(other_premise):
                delete_rule = True
                break
        if delete_rule:
            rule_vertices_to_delete.append(rule.vertex)

    subgraph_rule_vertices_to_delete: igraph.VertexSeq = subgraph.vs.select(
        original_ref_in=[vertex["original_ref"] for vertex in rule_vertices_to_delete]
    )
    vertices_targeted_for_removal = list(subgraph_rule_vertices_to_delete) + list(
        subgraph.vs.select(_degree=0)
    )
    # if output_dir is not None:
    #     plot_before_fyd_rule_deletion(
    #         subgraph,
    #         subgraph_layout,
    #         visual_style,
    #         vertices_targeted_for_removal,
    #         output_dir=output_dir,
    #     )
    knowledge_base.graph.delete_vertices(rule_vertices_to_delete)
    subgraph.delete_vertices(subgraph_rule_vertices_to_delete)
    subgraph.vs.select(_degree=0).delete()
    # refresh the fuzzy anchor vertices
    # fuzzy_anchors_vertices = [
    #     vertex
    #     for vertex in list(knowledge_base.graph.vs)
    #     if isinstance(vertex["item"], tuple) and vertex["input"]
    # ]
    # subgraph = induce_subgraph(fuzzy_anchors_vertices, knowledge_base)

    # delete unconnected vertices (some rule vertices can become unconnected)
    knowledge_base.graph.vs.select(_degree=0).delete()

    # if output_dir is not None:
    #     plot_after_fyd_rule_deletion(subgraph, visual_style, output_dir=output_dir)

    if verbose:
        premise_terms: List[Tuple[int, int]] = [
            v["item"]
            for v in knowledge_base.graph.vs.select(
                lambda v: isinstance(v["item"], tuple) and v["input"]
            )
        ]
        print(
            f"Only {len(premise_terms)} terms remain in the knowledge base after the FYD Method "
            f"(originally there were {len(fuzzy_anchors_vertices)} terms)."
        )
        print(f"Only {len(knowledge_base.rules)} rules remain.")

    # list(knowledge_base.get_granules(True))[0]["item"].plot()
    grouped_fuzzy_sets: FuzzySetGroup = knowledge_base.graph.vs.find(
        lambda v: isinstance(v["item"], FuzzySetGroup)
    )["item"]

    selected_terms = {
        term for rule in knowledge_base.rules for term in rule.premise.indices[0]
    }
    if output_dir is not None:
        grouped_fuzzy_sets.modules_list[0].plot(
            output_dir=output_dir, selected_terms=list(selected_terms)
        )

    return knowledge_base


def calc_fyd_heuristic(
    fuzzy_anchors_vertices,
    input_data,
    knowledge_base,
    subgraph,
    subgraph_fuzzy_anchors_vertices,
    device: torch.device,
):
    """
    Calculates the heuristic for the Frequent-Yet-Discernible Method.

    Args:
        fuzzy_anchors_vertices:
        input_data:
        knowledge_base:
        subgraph:
        subgraph_fuzzy_anchors_vertices:
        device: The device to use.

    Returns:

    """
    subgraph_rule_vertices = subgraph.vs.select(tags_eq={"premise", "relation"})

    # begin of heuristic calculation
    vertices_closeness = subgraph.closeness(
        subgraph_fuzzy_anchors_vertices, mode="all"  # mode 'all' uses undirected paths
    )  # possible nan values
    vertices_closeness = np.nan_to_num(vertices_closeness, nan=0.0)
    vertices_usage = np.array(
        subgraph.degree(subgraph_fuzzy_anchors_vertices, mode="out")
    ) / len(subgraph_rule_vertices)
    vertices_usage_closeness = vertices_usage * vertices_closeness
    vertices_usage_closeness = (
        vertices_usage_closeness - np.nanmin(vertices_usage_closeness)
    ) / (np.nanmax(vertices_usage_closeness) - np.nanmin(vertices_usage_closeness))
    (
        normalized_scalar_cardinalities,
        vertices_heuristics,
    ) = calc_normalized_scalar_cardinalities(
        fuzzy_anchors_vertices,
        input_data,
        knowledge_base,
        vertices_usage_closeness,
        device=device,
    )

    return normalized_scalar_cardinalities * vertices_heuristics

    # vertices_heuristics = normalized_scalar_cardinalities * vertices_heuristics
    # vertices_heuristics = yager_t_norm(
    #     normalized_scalar_cardinalities,
    #     vertices_heuristics,
    #     w_parameter=knowledge_base.config.fuzzy.t_norm.yager,
    # )
    # return vertices_heuristics


def calc_normalized_scalar_cardinalities(
    fuzzy_anchors_vertices,
    input_data,
    knowledge_base,
    vertices_usage_closeness,
    device: torch.device,
):
    """
    Part of the Frequent-Yet-Discernible heuristic. Calculates the normalized scalar
    cardinalities and the vertices heuristics. The vertices heuristics are calculated as
    1 - normalized usage closeness. The normalized scalar cardinalities are calculated as
    the scalar cardinalities divided by the maximum scalar cardinality.

    Args:
        fuzzy_anchors_vertices: The vertices of the fuzzy anchors.
        input_data: The input data.
        knowledge_base: The knowledge base.
        vertices_usage_closeness: The vertices' usage & closeness.
        device: The device to use.

    Returns:
        The normalized scalar cardinalities and the vertices heuristics.
    """
    # heuristic of vertices; higher is better
    vertices_heuristics = 1 - vertices_usage_closeness
    scalar_cardinalities = calc_scalar_cardinality(
        knowledge_base, fuzzy_anchors_vertices, input_data, device=device
    )  # this results in high values (e.g., 300+)
    normalized_scalar_cardinalities = (
        scalar_cardinalities - np.nanmin(scalar_cardinalities)
    ) / (np.nanmax(scalar_cardinalities) - np.nanmin(scalar_cardinalities))
    # normalized_scalar_cardinalities = np.power(
    #     np.nan_to_num(normalized_scalar_cardinalities, 0), np.e
    # )
    return normalized_scalar_cardinalities, vertices_heuristics


def induce_subgraph(fuzzy_anchors_vertices, knowledge_base):
    """
    Find the subgraph with the vertices of interest, which are the vertices called rule_vertices
    (i.e., the vertices that reference the fuzzy logic rule relationships) and the vertices that
    reference the fuzzy sets (i.e., fuzzy_anchors_vertices).

    Args:
        fuzzy_anchors_vertices:
        knowledge_base:

    Returns:

    """
    rule_vertices: igraph.VertexSeq = knowledge_base.select_by_tags(
        tags={"premise", "relation"}
    )
    rule_vertices["original_ref"] = list(range(len(rule_vertices)))
    all_vertices = sorted(
        list(set(fuzzy_anchors_vertices).union(rule_vertices)), key=lambda v: v.index
    )
    subgraph = knowledge_base.graph.induced_subgraph(all_vertices)
    subgraph.vs["original_ref"] = (
        [None] * len(fuzzy_anchors_vertices)
    ) + rule_vertices["original_ref"]
    return subgraph


def calc_scalar_cardinality(
    knowledge_base, fuzzy_anchors_vertices, input_data, device: torch.device
):
    """
    Calculate the 'scalar cardinality', which is summation of the input granulation membership
    across the input data.

    Args:
        knowledge_base: soft.computing.knowledge.KnowledgeBase
        fuzzy_anchors_vertices:
        input_data: The input data
        device: The device to use.

    Returns:

    """
    scalar_cardinality = knowledge_base.granulation_layers["input"](
        input_data.to(device=device)
    ).degrees.sum(dim=0)
    return np.array(
        [scalar_cardinality[vertex["item"]].item() for vertex in fuzzy_anchors_vertices]
    )


def find_heuristic_cutoff(vertices_heuristics):
    """
    Given the valid heuristics (i.e., not Numpy 'nan'), find the cutoff value using
    the 'Kneedle' algorithm.

    Args:
        vertices_heuristics: The heuristic value of each vertex.

    Returns:
        The heuristic cutoff
    """
    valid_heuristics = sorted([val for val in vertices_heuristics if val > 0])
    observation = range(len(valid_heuristics))
    # knee_value = KneeLocator(
    #     observation,
    #     valid_heuristics,
    #     curve="convex",
    #     direction="increasing",
    #     online=True,
    #     S=3.0,
    #     # interp_method="interp1d",
    #     interp_method="polynomial",
    #     polynomial_degree=2,
    # ).knee_y  # the heuristic cutoff
    knee_value = KneeLocator(
        observation,
        valid_heuristics,
        curve="convex",
        direction="increasing",
        online=True,
        # interp_method="interp1d",
        polynomial_degree=3,
    ).knee_y  # the heuristic cutoff

    if knee_value is None:
        knee_value = 0.0

    return knee_value

    # the below could be used for offline fuzzy RL
    # if knee_value is None:
    #     return np.mean(valid_heuristics)
    #
    # return np.mean(valid_heuristics)
    # # return min(np.mean(valid_heuristics), knee_value)
