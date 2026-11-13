from itertools import pairwise
from pathlib import Path
from textwrap import fill

from cyclopts import App
import huracanpy
from matplotlib.lines import Line2D
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Patch,
    Rectangle,
)
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wcsi import wcsi
from wcsi.plot.trackplot import natures as TRACK_NATURES


DATA_DIR = Path("/home/lsaffin/Documents/meteorology/data/huracan/tracks")
OUTPUT_FILE = Path("ibtracs_intensifying.csv")
PLOT_FILE = Path("ibtracs_intensifying_sankey.pdf")
PLOT_FILE_2007 = Path("ibtracs_intensifying_sankey_2007-2024.pdf")
FLOWCHART_FILE = Path("ibtracs_intensifying_flowchart.pdf")
TROPICAL_NATURE = "TS"
VALID_INTENSIFICATION_NATURES = ("TS", "MX", "NR", "DS")
INELIGIBLE_TRACK_NATURES = ("ET", "SS")
MAX_CONSECUTIVE_POINTS = 4
FLAT_INTENSIFICATION_THRESHOLD = -0.01
SANKEY_STAGE_SPACING = 0.58
SANKEY_STAGE_LABELS = {
    "total": "Total",
    "eligibility": "Track eligibility",
    "wind_nature_data": "Wind and nature data",
    "outcome": "Classification",
    "non_tc_category": "Category",
}
SANKEY_NODE_LABELS = {
    "All tracks": "All tracks",
    "Not enough points": "<1 day",
    "ET or SS nature": "ET/SS",
    "Remaining": "Remaining",
    "Not enough valid wind and nature data": "Insufficient",
    "Valid wind and nature data": "Sufficient",
    "Not intensifying": "Not intensifying",
    "TS/MX/NR/DS and flat": "TS/MX/NR/DS and flat",
    "TS/MX/NR/DS and intensifying": "TS/MX/NR/DS and intensifying",
    "TS and intensifying": "TS and intensifying",
}
SANKEY_CATEGORY_LABEL = "{category}"
SANKEY_NOT_INTENSIFYING_CATEGORY_LABEL = "Cat {category}"
SANKEY_OTHER_CATEGORY_LABEL = "Cat {category}"
SANKEY_CATEGORY_LABEL_OVERRIDES = {"-1": "TD", "0": "TS"}
SANKEY_NATURE_ORDER = ("TS", "MX", "NR", "DS", "SS", "ET")
SANKEY_LABEL_FONTSIZE = 7
SANKEY_MAX_NODE_GAP = 0.035
SANKEY_CLASSIFICATION_NODE_GAP = 0.06
SANKEY_CATEGORY_GROUP_GAP = 0.05
SANKEY_STAGE_GAP_BUDGET = 0.28
SANKEY_FLOW_HEIGHT = 0.60
SANKEY_MIN_NODE_HEIGHT = 0.02
SANKEY_CLASSIFICATION_WRAP = 18
SANKEY_RIGHT_STAGE_SHIFT = 0.0
SANKEY_WCSI_LEGEND_X = 0.34
SANKEY_LEGEND_Y = 0.15
WCSI_ALPHA = 1.0
NON_WCSI_ALPHA = 0.7

app = App()


@app.default
def main(
    *,
    summary_file: Path = DATA_DIR / "WCSI_summary_all.parquet",
    tracks_file: Path = DATA_DIR / "IBTrACS.nc",
    output_file: Path = OUTPUT_FILE,
    plot_file: Path = PLOT_FILE,
    plot_file_2007: Path = PLOT_FILE_2007,
    flowchart_file: Path = FLOWCHART_FILE,
    saved_summary: Path | None = None,
):
    """Classify IBTrACS intensification or report an existing saved result.

    Parameters
    ----------
    summary_file
        WCSI summary parquet used to select and describe IBTrACS tracks.
    tracks_file
        IBTrACS track file used when calculating new classifications.
    output_file
        CSV written after calculating new classifications.
    plot_file
        Proportional 1979--2024 classification Sankey plot to create.
    plot_file_2007
        Proportional 2007--2024 classification Sankey plot to create.
    flowchart_file
        Portrait node-based classification flowchart to create.
    saved_summary
        Existing CSV produced by this script. When supplied, skip track preparation
        and classification and only print its summary.
    """
    if saved_summary is not None:
        comparison = print_summary(pd.read_csv(saved_summary))
        comparison_1979 = subset_years(comparison, 1979, 2024)
        plot_summary(comparison_1979, plot_file)
        plot_summary(subset_years(comparison, 2007, 2024), plot_file_2007)
        plot_flowchart(comparison_1979, flowchart_file)
        return

    summary = pd.read_parquet(summary_file)
    summary = summary.loc[summary.id_ibtracs != ""].copy()

    ibtracs = huracanpy.load(str(tracks_file))[["track_id", "time", "wind", "nature"]]
    ibtracs = ibtracs.hrcn.sel_id(summary.id_ibtracs.unique())
    results = classify_tracks(ibtracs, summary.id_ibtracs.unique())
    comparison = print_summary(summary, results, output_file=output_file)
    comparison_1979 = subset_years(comparison, 1979, 2024)
    plot_summary(comparison_1979, plot_file)
    plot_summary(subset_years(comparison, 2007, 2024), plot_file_2007)
    plot_flowchart(comparison_1979, flowchart_file)


def subset_years(summary, start_year, end_year):
    """Return tracks whose IBTrACS ID year lies in the inclusive period."""
    year = pd.to_numeric(summary.id_ibtracs.str[:4], errors="coerce")
    return summary.loc[year.between(start_year, end_year)].copy()


def classify_tracks(ibtracs, track_ids):
    """Prepare tracks, then classify TS and mixed-nature wind runs."""
    results, batches = prepare_tracks(ibtracs)
    missing = set(track_ids) - set(results.id_ibtracs)
    results = pd.concat(
        [
            results,
            pd.DataFrame(
                classification_result(track_id, status="no six-hourly points")
                for track_id in missing
            ),
        ],
        ignore_index=True,
    )

    strict_ids = {selection: set() for selection in batches}
    flat_ids = {selection: set() for selection in batches}
    for selection, intensity in batches.items():
        intensity = intensity.expand_dims(pressure=[850])
        track_id = intensity.track_id.values

        strict = wcsi(
            track_id,
            relative_vorticity=intensity,
            vort_threshold=None,
            npoints=MAX_CONSECUTIVE_POINTS,
        )
        flat = wcsi(
            track_id,
            relative_vorticity=intensity,
            vort_threshold=None,
            intensification_threshold=FLAT_INTENSIFICATION_THRESHOLD,
            npoints=MAX_CONSECUTIVE_POINTS,
        )
        source_track_id = intensity.source_track_id.values
        strict_ids[selection].update(np.unique(source_track_id[strict]))
        flat_ids[selection].update(np.unique(source_track_id[flat]))

    results["istc_ts"] = results.id_ibtracs.isin(strict_ids.get("ts", set()))
    results["istc"] = (
        results.id_ibtracs.isin(strict_ids.get("valid_nature", set())) | results.istc_ts
    )
    results["istc_flat"] = (
        results.id_ibtracs.isin(flat_ids.get("valid_nature", set())) | results.istc
    )
    return results


def prepare_tracks(ibtracs):
    """Describe wind completeness and prepare valid-nature wind runs."""
    track_id, inverse, record_count = np.unique(
        ibtracs.track_id.values, return_inverse=True, return_counts=True
    )
    ntracks = len(track_id)

    tropical = ibtracs.nature.values == TROPICAL_NATURE
    valid_nature = np.isin(ibtracs.nature.values, VALID_INTENSIFICATION_NATURES)
    wind_valid = ibtracs.wind.notnull().values
    wind_count = np.bincount(inverse, weights=wind_valid, minlength=ntracks).astype(int)
    wind_run, _, _ = consecutive_run_info(wind_valid, inverse, track_id, "wind")
    ts_run, ts_records, _ = consecutive_run_info(
        tropical & wind_valid, inverse, track_id, "ts-wind"
    )
    nature_wind_run, nature_wind_records, _ = consecutive_run_info(
        valid_nature & wind_valid, inverse, track_id, "valid-nature-wind"
    )

    wind_data = np.full(ntracks, "insufficient valid wind", dtype=object)
    wind_data[wind_count == 0] = "no valid wind"
    wind_data[wind_run >= MAX_CONSECUTIVE_POINTS] = "enough valid wind"
    wind_data[wind_count == record_count] = "fully valid wind"
    nature_wind_sufficient = nature_wind_run >= MAX_CONSECUTIVE_POINTS
    status = np.where(
        nature_wind_sufficient, "classified", "insufficient nature and wind data"
    )

    results = pd.DataFrame(
        {
            "id_ibtracs": track_id,
            "status": status,
            "wind_data": wind_data,
            "nature_wind_sufficient": nature_wind_sufficient,
            "wind_npoints": wind_count,
            "wind_run_npoints": np.minimum(MAX_CONSECUTIVE_POINTS, wind_run),
            "nature_wind_run_npoints": np.minimum(
                MAX_CONSECUTIVE_POINTS, nature_wind_run
            ),
        }
    )

    batches = {}
    for selection, records, eligible in (
        ("ts", ts_records, ts_run >= MAX_CONSECUTIVE_POINTS),
        ("valid_nature", nature_wind_records, nature_wind_sufficient),
    ):
        selected = records & eligible[inverse]
        if not selected.any():
            continue
        segment_ids = consecutive_run_info(selected, inverse, track_id, selection)[2][
            selected
        ]
        batches[selection] = ibtracs.wind.isel(record=selected).assign_coords(
            track_id=("record", segment_ids),
            source_track_id=("record", ibtracs.track_id.values[selected]),
        )

    return results, batches


def consecutive_run_info(valid, inverse, track_ids, prefix):
    """Return maximum run length per track and a unique ID for every valid run."""
    same_track = np.r_[False, inverse[1:] == inverse[:-1]]
    starts = valid & ~(np.r_[False, valid[:-1]] & same_track)
    run_number = np.cumsum(starts) - 1
    run_lengths = np.bincount(run_number[valid])
    run_tracks = inverse[np.flatnonzero(starts)]

    max_length = np.zeros(len(track_ids), dtype=int)
    np.maximum.at(max_length, run_tracks, run_lengths)

    segment_ids = np.full(len(valid), "", dtype=object)
    segment_ids[valid] = np.array(
        [
            f"{prefix}:{track_ids[track]}:{run:08d}"
            for run, track in enumerate(run_tracks)
        ],
        dtype=object,
    )[run_number[valid]]
    return max_length, valid, segment_ids


def classification_result(track_id, *, status):
    return {
        "id_ibtracs": track_id,
        "status": status,
        "wind_data": "no valid wind",
        "nature_wind_sufficient": False,
        "wind_npoints": 0,
        "wind_run_npoints": 0,
        "nature_wind_run_npoints": 0,
    }


def print_summary(summary, results=None, *, output_file=None):
    """Print each classification stage and its summary-data breakdown."""
    if results is None:
        comparison = summary.copy()
    else:
        # An IBTrACS cyclone may be matched to more than one reanalysis track.
        wcsi_by_track = summary.groupby("id_ibtracs", as_index=False).agg(
            WCSI=("WCSI", "any"),
            ibtracs_nature=("ibtracs_nature", "first"),
            ibtracs_points_6h=("ibtracs_points_6h", "max"),
            ibtracs_category=("ibtracs_category", "max"),
        )
        comparison = wcsi_by_track.merge(results, on="id_ibtracs", how="left")
    comparison["ibtracs_nature"] = comparison.ibtracs_nature.replace("", "unrecorded")
    comparison["ibtracs_category"] = comparison.ibtracs_category.astype("Int64")
    ineligible_nature = comparison.ibtracs_nature.isin(INELIGIBLE_TRACK_NATURES)
    comparison.loc[ineligible_nature, "nature_wind_sufficient"] = False
    for column in ("istc_ts", "istc", "istc_flat"):
        if column in comparison:
            comparison.loc[ineligible_nature, column] = False
    if "status" in comparison:
        comparison.loc[ineligible_nature, "status"] = "ineligible track nature"

    print_breakdown(
        "Total number of unique IBTrACS tracks",
        comparison,
    )
    plot_data, _, _ = prepare_plot_data(comparison)
    for stage, labels in (
        ("eligibility", ["Not enough points", "ET or SS nature", "Remaining"]),
        (
            "wind_data_group",
            [
                "Fully valid wind data",
                "Four-point valid wind run",
                "Not enough valid wind data",
                "No valid wind data",
            ],
        ),
        (
            "nature_wind_group",
            ["Enough valid nature and wind data", "Not enough nature and wind data"],
        ),
        (
            "outcome",
            [
                "TS and intensifying",
                "TS/MX/NR/DS and intensifying",
                "TS/MX/NR/DS and flat",
                "Not intensifying",
            ],
        ),
    ):
        for label in labels:
            subset = plot_data[plot_data[stage] == label]
            if not subset.empty:
                print_breakdown(label, subset, include_results=stage == "outcome")

    if output_file is not None:
        comparison.to_csv(output_file, index=False)
        print(f"\nSaved per-track results to {output_file}")
    return comparison


def print_breakdown(title, tracks, *, include_results=False):
    """Print one decision stage by nature, category, and WCSI classification."""
    print(f"\n{title}: {len(tracks):,}")
    if tracks.empty:
        return

    groups = tracks.groupby(
        ["ibtracs_nature", "ibtracs_category", "WCSI"], dropna=False, observed=True
    )
    if include_results:
        table = groups.agg(
            tracks=("id_ibtracs", "size"),
            ts_intensifying=("istc_ts", "sum"),
            intensifying=("istc", "sum"),
            flat_or_intensifying=("istc_flat", "sum"),
        )
    else:
        table = groups.size().rename("tracks")
    print(table.to_string())


def plot_summary(summary, filename):
    """Plot a proportional Sankey of the classification decisions."""
    data, stages, orders = prepare_plot_data(summary)
    data, stages, orders, display_labels = prepare_sankey_tree(data, stages, orders)
    nature_colors = nature_color_map(data.ibtracs_nature.unique())
    positions, scale = sankey_positions(data, stages, orders)

    fig, ax = plt.subplots(figsize=(8, 5))
    for left, right in pairwise(stages):
        draw_sankey_links(
            ax, data, left, right, positions, scale, nature_colors, orders
        )
    draw_sankey_nodes(
        ax,
        data,
        stages,
        orders,
        positions,
        scale,
        nature_colors,
        display_labels,
    )
    add_sankey_legends(ax, nature_colors)

    top = max(position[2] for position in positions.values())
    bottom = min(position[1] for position in positions.values())
    title_y = top + 0.045
    stage_x = sankey_stage_positions(stages)
    for x, stage in zip(stage_x, stages, strict=True):
        title = SANKEY_STAGE_LABELS[stage.removeprefix("tree_")]
        ax.text(x, title_y, title, ha="center", va="bottom", fontsize=7, weight="bold")
    ax.set(
        xlim=(-0.1, stage_x[-1] + 0.55),
        ylim=(bottom - 0.025, title_y + 0.025),
    )
    ax.axis("off")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved proportional Sankey plot to {filename}")


def prepare_sankey_tree(data, stages, orders):
    """Create Sankey nodes with one combined wind-and-nature data stage."""
    data = data.copy()
    data["wind_nature_data"] = pd.NA
    no_wind = data.wind_data_group == "No valid wind data"
    insufficient = (data.wind_data_group == "Not enough valid wind data") | (
        data.nature_wind_group == "Not enough nature and wind data"
    )
    sufficient = data.nature_wind_group == "Enough valid nature and wind data"
    data.loc[no_wind | insufficient, "wind_nature_data"] = (
        "Not enough valid wind and nature data"
    )
    data.loc[sufficient, "wind_nature_data"] = "Valid wind and nature data"

    wind_stage = stages.index("wind_data_group")
    stages = [
        *stages[:wind_stage],
        "wind_nature_data",
        *stages[stages.index("nature_wind_group") + 1 :],
    ]
    orders = {
        **orders,
        "wind_nature_data": [
            "Valid wind and nature data",
            "Not enough valid wind and nature data",
        ],
    }
    tree = data.copy()
    tree_stages = [f"tree_{stage}" for stage in stages]
    tree_orders = {stage: [] for stage in tree_stages}
    display_labels = {}
    paths = [()] * len(tree)

    for stage, tree_stage in zip(stages, tree_stages, strict=True):
        node_ids = []
        for path, label in zip(paths, tree[stage], strict=True):
            if path is None or pd.isna(label):
                node_ids.append(None)
                continue
            if stage == "non_tc_category":
                outcome = next(
                    (
                        path_label
                        for path_stage, path_label in path
                        if path_stage == "outcome"
                    ),
                    None,
                )
                if outcome == "Not intensifying":
                    node_id = (*path, (stage, label))
                    display_label = f"not_intensifying:{label}"
                else:
                    node_id = ((stage, f"other:{label}"),)
                    display_label = f"other:{label}"
            else:
                node_id = (*path, (stage, label))
                display_label = label
            node_ids.append(node_id)
            display_labels[node_id] = display_label
        tree[tree_stage] = pd.Series(node_ids, index=tree.index, dtype=object)
        paths = node_ids

    # Preserve recursive path order so every downstream branch stays attached to
    # its combined data-quality node.
    for node in build_flowchart_tree(data, stages, orders):
        tree_stage = f"tree_{node['stage']}"
        if node["stage"] == "non_tc_category":
            outcome = next(
                (label for stage, label in node["id"] if stage == "outcome"),
                None,
            )
            if outcome == "Not intensifying":
                node_id = node["id"]
            else:
                node_id = ((node["stage"], f"other:{node['label']}"),)
        else:
            node_id = node["id"]
        if node_id not in tree_orders[tree_stage]:
            tree_orders[tree_stage].append(node_id)

    # sankey_positions lays nodes out from bottom to top. Keep incomplete and
    # invalid branches low, with the strongest valid branch at the top.
    bottom_to_top = {
        "eligibility": [
            "Not enough points",
            "ET or SS nature",
            "Remaining",
        ],
        "wind_nature_data": [
            "Not enough valid wind and nature data",
            "Valid wind and nature data",
        ],
        "outcome": [
            "Not intensifying",
            "TS/MX/NR/DS and flat",
            "TS/MX/NR/DS and intensifying",
            "TS and intensifying",
        ],
    }
    for stage, labels in bottom_to_top.items():
        tree_stage = f"tree_{stage}"
        rank = {label: index for index, label in enumerate(labels)}
        tree_orders[tree_stage].sort(key=lambda node_id: rank[display_labels[node_id]])

    category_stage = "tree_non_tc_category"
    tree_orders[category_stage].sort(
        key=lambda node_id: sankey_category_bottom_to_top_key(display_labels[node_id])
    )
    return tree, tree_stages, tree_orders, display_labels


def prepare_plot_data(summary):
    """Create the common hierarchy used by both classification figures."""
    data = summary.copy()
    for column in ("istc_ts", "istc", "istc_flat", "nature_wind_sufficient"):
        data[column] = data[column].fillna(False).astype(bool)
    data["wind_data"] = data.wind_data.fillna("no valid wind")
    enough_points = data.ibtracs_points_6h >= MAX_CONSECUTIVE_POINTS
    ineligible_nature = data.ibtracs_nature.isin(INELIGIBLE_TRACK_NATURES)
    remaining = enough_points & ~ineligible_nature

    data["eligibility"] = "Not enough points"
    data.loc[enough_points & ineligible_nature, "eligibility"] = "ET or SS nature"
    data.loc[remaining, "eligibility"] = "Remaining"

    data["wind_data_group"] = pd.NA
    wind_labels = {
        "fully valid wind": "Fully valid wind data",
        "enough valid wind": "Four-point valid wind run",
        "insufficient valid wind": "Not enough valid wind data",
        "no valid wind": "No valid wind data",
    }
    data.loc[remaining, "wind_data_group"] = data.loc[remaining, "wind_data"].map(
        wind_labels
    )

    wind_eligible = remaining & data.wind_data.isin(
        ["fully valid wind", "enough valid wind"]
    )
    data["nature_wind_group"] = pd.NA
    data.loc[wind_eligible, "nature_wind_group"] = np.where(
        data.loc[wind_eligible, "nature_wind_sufficient"],
        "Enough valid nature and wind data",
        "Not enough nature and wind data",
    )

    data["outcome"] = pd.NA
    classifiable = wind_eligible & data.nature_wind_sufficient
    data.loc[classifiable & data.istc_ts, "outcome"] = "TS and intensifying"
    data.loc[classifiable & ~data.istc_ts & data.istc, "outcome"] = (
        "TS/MX/NR/DS and intensifying"
    )
    data.loc[
        classifiable & ~data.istc_ts & ~data.istc & data.istc_flat,
        "outcome",
    ] = "TS/MX/NR/DS and flat"
    data.loc[
        classifiable & ~data.istc_ts & ~data.istc & ~data.istc_flat,
        "outcome",
    ] = "Not intensifying"
    data["non_tc_category"] = pd.NA
    categories = (
        data.ibtracs_category.astype("Int64").astype(str).replace("<NA>", "Unknown")
    )
    data.loc[data.outcome.notna(), "non_tc_category"] = "Cat " + categories

    stages = [
        "total",
        "eligibility",
        "wind_data_group",
        "nature_wind_group",
        "outcome",
        "non_tc_category",
    ]
    data["total"] = "All tracks"
    orders = {
        "total": ["All tracks"],
        "eligibility": [
            "Not enough points",
            "ET or SS nature",
            "Remaining",
        ],
        "wind_data_group": [
            "Fully valid wind data",
            "Four-point valid wind run",
            "Not enough valid wind data",
            "No valid wind data",
        ],
        "nature_wind_group": [
            "Enough valid nature and wind data",
            "Not enough nature and wind data",
        ],
        "outcome": [
            "TS and intensifying",
            "TS/MX/NR/DS and intensifying",
            "TS/MX/NR/DS and flat",
            "Not intensifying",
        ],
        "non_tc_category": sorted(
            data.non_tc_category.dropna().unique(), key=category_sort_key
        ),
    }
    return data, stages, orders


def plot_flowchart(summary, filename):
    """Plot every classification path as a compact portrait tree."""
    data, stages, orders = prepare_plot_data(summary)
    stage_titles = {
        "total": "Total",
        "eligibility": "Track eligibility",
        "wind_data_group": "Wind data",
        "nature_wind_group": "Nature and wind data",
        "outcome": "Classification",
        "non_tc_category": "Category",
    }
    nodes = build_flowchart_tree(data, stages, orders)
    positions = flowchart_positions(nodes)

    fig, ax = plt.subplots(figsize=(7.5, 9.0))
    draw_flowchart_links(ax, nodes, positions)
    draw_flowchart_nodes(ax, nodes, positions)
    stage_x = dict(zip(stages, np.linspace(0.07, 0.93, len(stages)), strict=True))
    for stage, title in stage_titles.items():
        ax.text(
            stage_x[stage],
            1.008,
            title,
            ha="center",
            va="bottom",
            fontsize=7.2,
            weight="bold",
        )
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.01, 1.04))
    ax.axis("off")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved portrait classification flowchart to {filename}")


def build_flowchart_tree(data, stages, orders):
    """Build path-specific nodes, duplicating labels reached by different parents."""
    nodes = []

    def add_children(subset, depth, path, parent):
        stage = stages[depth]
        present = set(subset[stage].dropna())
        for label in orders[stage]:
            if label not in present:
                continue
            child_data = subset[subset[stage] == label]
            node_id = (*path, (stage, label))
            node = {
                "id": node_id,
                "parent": parent,
                "stage": stage,
                "label": label,
                "count": len(child_data),
                "children": [],
            }
            nodes.append(node)
            if parent is not None:
                parent["children"].append(node)
            if depth + 1 < len(stages) and child_data[stages[depth + 1]].notna().any():
                add_children(child_data, depth + 1, node_id, node)

    add_children(data, 0, (), None)
    return nodes


def flowchart_positions(nodes):
    """Use terminal paths as vertical slots and center parents on their children."""
    positions = {}
    leaves = [node for node in nodes if not node["children"]]
    leaf_y = dict(
        zip(
            (node["id"] for node in leaves),
            np.linspace(0.975, 0.025, len(leaves)),
            strict=True,
        )
    )
    stages = list(dict.fromkeys(node["stage"] for node in nodes))
    stage_x = dict(zip(stages, np.linspace(0.06, 0.94, len(stages)), strict=True))

    def position(node):
        if node["id"] in positions:
            return positions[node["id"]]
        if node["children"]:
            y = np.mean([position(child)[1] for child in node["children"]])
        else:
            y = leaf_y[node["id"]]
        positions[node["id"]] = (stage_x[node["stage"]], y)
        return positions[node["id"]]

    position(nodes[0])
    return positions


def draw_flowchart_links(ax, nodes, positions):
    """Connect each path-specific node only to its own parent."""
    maximum = max(node["count"] for node in nodes)
    for node in nodes:
        if node["parent"] is None:
            continue
        x0, y0 = positions[node["parent"]["id"]]
        x1, y1 = positions[node["id"]]
        width = 0.35 + 1.15 * np.sqrt(node["count"] / maximum)
        ax.add_patch(
            FancyArrowPatch(
                (x0 + 0.073, y0),
                (x1 - 0.073, y1),
                arrowstyle="-|>",
                mutation_scale=5,
                color="#999999",
                alpha=0.6,
                linewidth=width,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )


def draw_flowchart_nodes(ax, nodes, positions):
    """Draw labelled count nodes with colors keyed to decision meaning."""
    default_colors = {
        "total": "#d9e2ec",
        "eligibility": "#f2d4c9",
        "wind_data_group": "#cfe2f3",
        "nature_wind_group": "#d9ead3",
        "outcome": "#eadcf0",
        "non_tc_category": "#e4d8ea",
    }
    semantic_colors = {
        "TS and intensifying": "#6aa84f",
        "TS/MX/NR/DS and intensifying": "#93c47d",
        "TS/MX/NR/DS and flat": "#ffe599",
        "Not intensifying": "#e6b8af",
    }
    for node in nodes:
        x, y = positions[node["id"]]
        width = 0.15
        height = 0.026
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                facecolor=semantic_colors.get(
                    node["label"], default_colors[node["stage"]]
                ),
                edgecolor="#444444",
                linewidth=0.55,
                zorder=3,
            )
        )
        ax.text(
            x,
            y,
            f"{node['label']}  {node['count']:,}",
            ha="center",
            va="center",
            fontsize=5.3,
            zorder=4,
        )


def nature_color_map(natures):
    mapping = TRACK_NATURES["ibtracs"]
    present = set(natures)
    colors = {
        nature: mapping[nature][1]
        for nature in SANKEY_NATURE_ORDER
        if nature in present
    }
    colors.update(
        {nature: "#666666" for nature in sorted(present - set(SANKEY_NATURE_ORDER))}
    )
    return colors


def category_sort_key(label):
    value = label.removeprefix("Cat ")
    return (value == "Unknown", float(value) if value != "Unknown" else np.inf)


def category_bottom_to_top_key(label):
    """Put unknown categories lowest and stronger categories progressively higher."""
    value = label.removeprefix("Cat ")
    return (value != "Unknown", float(value) if value != "Unknown" else -np.inf)


def sankey_category_bottom_to_top_key(label):
    """Order separate not-intensifying and combined-other category groups."""
    group, category = label.split(":", maxsplit=1)
    return (group == "other", *category_bottom_to_top_key(category))


def sankey_node_label(label):
    """Translate an internal branch value to independently configurable text."""
    if label.startswith("not_intensifying:"):
        category = label.removeprefix("not_intensifying:Cat ")
        return sankey_category_label(SANKEY_NOT_INTENSIFYING_CATEGORY_LABEL, category)
    if label.startswith("other:"):
        category = label.removeprefix("other:Cat ")
        return sankey_category_label(SANKEY_OTHER_CATEGORY_LABEL, category)
    if label.startswith("Cat "):
        return sankey_category_label(SANKEY_CATEGORY_LABEL, label.removeprefix("Cat "))
    return SANKEY_NODE_LABELS.get(label, label)


def sankey_category_label(template, category):
    """Format a category label, applying configured whole-label overrides."""
    return SANKEY_CATEGORY_LABEL_OVERRIDES.get(
        category, template.format(category=category)
    )


def sankey_stage_positions(stages):
    """Return equally spaced horizontal stage positions."""
    return [index * SANKEY_STAGE_SPACING for index in range(len(stages))]


def sankey_positions(data, stages, orders):
    minimum_height = SANKEY_MIN_NODE_HEIGHT
    available_height = 0.82
    stage_gaps = {
        stage: min(
            (
                SANKEY_CLASSIFICATION_NODE_GAP
                if stage.endswith("outcome")
                else SANKEY_MAX_NODE_GAP
            ),
            SANKEY_STAGE_GAP_BUDGET / max(1, len(orders[stage]) - 1),
        )
        for stage in stages
    }

    # Find one common scale so ribbon widths stay comparable across every stage,
    # including stages containing many duplicated path-specific nodes.
    scale = SANKEY_FLOW_HEIGHT / len(data)
    while True:
        required = []
        for stage in stages:
            counts = data[stage].value_counts()
            heights = [max(count * scale, minimum_height) for count in counts]
            required.append(sum(heights) + stage_gaps[stage] * max(0, len(heights) - 1))
        if max(required, default=0) <= available_height:
            break
        scale *= 0.96

    positions = {}
    for x, stage in zip(sankey_stage_positions(stages), stages, strict=True):
        gap = stage_gaps[stage]
        counts = data[stage].value_counts()
        labels = [label for label in orders[stage] if label in counts]
        node_heights = {
            label: max(counts[label] * scale, minimum_height) for label in labels
        }
        height = sum(node_heights.values()) + gap * (len(labels) - 1)
        y = 1 - height
        for label in labels:
            node_height = node_heights[label]
            positions[(stage, label)] = (
                x,
                y,
                y + node_height,
            )
            y += node_height + gap

    category_stage = next(
        (stage for stage in stages if stage.endswith("non_tc_category")), None
    )
    outcome_stage = next((stage for stage in stages if stage.endswith("outcome")), None)
    if category_stage is not None and outcome_stage is not None:
        outcome_keys = [key for key in positions if key[0] == outcome_stage]
        for key in outcome_keys:
            x, bottom, top = positions[key]
            positions[key] = (
                x,
                bottom + SANKEY_RIGHT_STAGE_SHIFT,
                top + SANKEY_RIGHT_STAGE_SHIFT,
            )
        other_category_keys = [
            key
            for key in positions
            if key[0] == category_stage and key[1][-1][1].startswith("other:")
        ]
        not_category_keys = [
            key
            for key in positions
            if key[0] == category_stage
            and any(
                stage == "outcome" and label == "Not intensifying"
                for stage, label in key[1]
            )
        ]
        ts_key = next(
            (key for key in outcome_keys if key[1][-1][1] == "TS and intensifying"),
            None,
        )
        not_key = next(
            (key for key in outcome_keys if key[1][-1][1] == "Not intensifying"),
            None,
        )
        if other_category_keys and ts_key is not None:
            cat_5_key = next(
                (key for key in other_category_keys if key[1][-1][1] == "other:Cat 5"),
                max(other_category_keys, key=lambda key: positions[key][2]),
            )
            shift = positions[ts_key][2] - positions[cat_5_key][2]
            for key in other_category_keys:
                x, bottom, top = positions[key]
                positions[key] = (x, bottom + shift, top + shift)
        if not_category_keys and not_key is not None:
            cat_2_key = next(
                (key for key in not_category_keys if key[1][-1][1] == "Cat 2"),
                max(not_category_keys, key=lambda key: positions[key][2]),
            )
            shift = positions[not_key][2] - positions[cat_2_key][2]
            for key in not_category_keys:
                x, bottom, top = positions[key]
                positions[key] = (x, bottom + shift, top + shift)
        if other_category_keys and not_category_keys and not_key is not None:
            upper_bottom = min(positions[key][1] for key in other_category_keys)
            lower_top = max(positions[key][2] for key in not_category_keys)
            extra_gap = SANKEY_CATEGORY_GROUP_GAP - (upper_bottom - lower_top)
            if extra_gap > 0:
                x, bottom, top = positions[not_key]
                positions[not_key] = (
                    x,
                    bottom - extra_gap,
                    top - extra_gap,
                )
                for key in not_category_keys:
                    x, bottom, top = positions[key]
                    positions[key] = (
                        x,
                        bottom - extra_gap,
                        top - extra_gap,
                    )
    return positions, scale


def draw_sankey_links(ax, data, left, right, positions, scale, colors, orders):
    subset = data.dropna(subset=[left, right])
    if subset.empty:
        return
    flows = (
        subset.groupby([left, right, "WCSI", "ibtracs_nature"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    left_counts = data[left].value_counts()
    right_counts = data[right].value_counts()
    left_offset = {
        key: value[1] + ((value[2] - value[1]) - left_counts[key[1]] * scale) / 2
        for key, value in positions.items()
        if key[0] == left
    }
    right_offset = {
        key: value[1] + ((value[2] - value[1]) - right_counts[key[1]] * scale) / 2
        for key, value in positions.items()
        if key[0] == right
    }
    flows["source_order"] = flows[left].map(
        {label: n for n, label in enumerate(orders[left])}
    )
    flows["target_order"] = flows[right].map(
        {label: n for n, label in enumerate(orders[right])}
    )
    flows["wcsi_order"] = flows.WCSI.map({False: 0, True: 1})
    flows["nature_order"] = flows.ibtracs_nature.map(
        {nature: n for n, nature in enumerate(reversed(colors))}
    )
    source_y = {}
    source_sorted = flows.sort_values(
        ["source_order", "target_order", "wcsi_order", "nature_order"]
    )
    for row in source_sorted.itertuples(index=True):
        key = (left, getattr(row, left))
        source_y[row.Index] = left_offset[key]
        left_offset[key] += row.count * scale

    target_y = {}
    target_sorted = flows.sort_values(
        ["target_order", "source_order", "wcsi_order", "nature_order"]
    )
    for row in target_sorted.itertuples(index=True):
        key = (right, getattr(row, right))
        target_y[row.Index] = right_offset[key]
        right_offset[key] += row.count * scale

    for row in flows.itertuples(index=True):
        source = getattr(row, left)
        target = getattr(row, right)
        nature = row.ibtracs_nature
        wcsi_class = row.WCSI
        height = row.count * scale
        source_key = (left, source)
        target_key = (right, target)
        y0 = source_y[row.Index]
        y1 = target_y[row.Index]
        add_ribbon(
            ax,
            positions[source_key][0] + 0.035,
            positions[target_key][0] - 0.035,
            y0,
            y1,
            height,
            colors[nature],
            WCSI_ALPHA if wcsi_class else NON_WCSI_ALPHA,
        )


def add_ribbon(ax, x0, x1, y0, y1, height, color, alpha):
    control = (x1 - x0) * 0.45
    vertices = [
        (x0, y0),
        (x0 + control, y0),
        (x1 - control, y1),
        (x1, y1),
        (x1, y1 + height),
        (x1 - control, y1 + height),
        (x0 + control, y0 + height),
        (x0, y0 + height),
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(vertices, codes), color=color, alpha=alpha, lw=0))


def draw_sankey_nodes(
    ax, data, stages, orders, positions, scale, colors, display_labels=None
):
    width = 0.07
    display_labels = display_labels or {}
    for stage in stages:
        for label in orders[stage]:
            key = (stage, label)
            if key not in positions:
                continue
            x, bottom, top = positions[key]
            node_data = data[data[stage] == label]
            y = bottom
            node_scale = (top - bottom) / len(node_data)
            counts = node_data.groupby(["WCSI", "ibtracs_nature"], observed=True).size()
            for wcsi_class in (False, True):
                # Nodes are drawn bottom-up, so reverse the legend order to place
                # Tropical (the first legend entry) at the top of each stack.
                for nature in reversed(colors):
                    count = counts.get((wcsi_class, nature), 0)
                    if not count:
                        continue
                    height = count * node_scale
                    ax.add_patch(
                        Rectangle(
                            (x - width / 2, y),
                            width,
                            height,
                            facecolor=colors[nature],
                            alpha=WCSI_ALPHA if wcsi_class else NON_WCSI_ALPHA,
                            edgecolor="none",
                            zorder=3,
                        )
                    )
                    y += height
            ax.add_patch(
                Rectangle(
                    (x - width / 2, bottom),
                    width,
                    top - bottom,
                    fill=False,
                    edgecolor="#333333",
                    linewidth=0.6,
                    zorder=4,
                )
            )
            node_label = sankey_node_label(display_labels.get(label, label))
            if stage.endswith("outcome"):
                node_label = fill(node_label, width=SANKEY_CLASSIFICATION_WRAP)
            ax.text(
                x + 0.055,
                (bottom + top) / 2,
                f"{node_label}\n{len(node_data):,}",
                ha="left",
                va="center",
                fontsize=SANKEY_LABEL_FONTSIZE,
                linespacing=1.0,
                zorder=5,
            )


def add_sankey_legends(ax, colors):
    mapping = TRACK_NATURES["ibtracs"]
    nature_handles = [
        Patch(
            color=color,
            label=mapping.get(nature, ("Unrecorded",))[0],
        )
        for nature, color in colors.items()
    ]
    nature_legend = ax.legend(
        handles=nature_handles,
        title="IBTrACS nature",
        ncols=2,
        loc="upper left",
        bbox_to_anchor=(0, SANKEY_LEGEND_Y),
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#777777",
        framealpha=1,
        fontsize=7,
        title_fontsize=8,
    )
    ax.add_artist(nature_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#555555", lw=6, alpha=WCSI_ALPHA, label="WCSI"),
            Line2D(
                [0],
                [0],
                color="#555555",
                lw=6,
                alpha=NON_WCSI_ALPHA,
                label="Not WCSI",
            ),
        ],
        title="ERA5 classification",
        loc="upper left",
        bbox_to_anchor=(SANKEY_WCSI_LEGEND_X, SANKEY_LEGEND_Y),
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#777777",
        framealpha=1,
        fontsize=7,
        title_fontsize=8,
    )


if __name__ == "__main__":
    app()
