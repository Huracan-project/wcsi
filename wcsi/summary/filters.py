"""Filter WCSI matching summaries by analysis period and verification category."""

import numpy as np
import pandas as pd


def year(
    table: pd.DataFrame,
    start_year: int,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Select summary rows within an inclusive range of years.

    The year of a matched observed cyclone is read from the first four characters of
    its ``id_ibtracs`` identifier. Rows without an IBTrACS identifier use the year of
    ``storm_start`` instead.

    Parameters
    ----------
    table
        WCSI matching summary. It must contain ``id_ibtracs`` and ``storm_start``;
        empty strings in ``id_ibtracs`` identify rows without an IBTrACS match.
    start_year
        First year to retain, inclusive.
    end_year
        Last year to retain, inclusive. If omitted, no upper bound is applied.

    Returns
    -------
    pandas.DataFrame
        Filtered table. Original index labels are preserved.

    Notes
    -----
    The input table is not modified. Non-empty IBTrACS identifiers are expected to
    start with a four-digit year, and ``storm_start`` must have a datetime-like
    pandas dtype.
    """

    # Select by ibtracs year
    early_ibtracs = (
        table[table.id_ibtracs != ""].id_ibtracs.str.slice(0, 4).astype(int)
        < start_year
    )
    table = table.drop(early_ibtracs[early_ibtracs].index)

    # For rows without IBTrACS, take the start of ERA5 track
    early_era5 = table[table.id_ibtracs == ""].storm_start.dt.year < start_year
    table = table.drop(early_era5[early_era5].index)

    if end_year is not None:
        late_ibtracs = (
            table[table.id_ibtracs != ""].id_ibtracs.str.slice(0, 4).astype(int)
            > end_year
        )
        table = table.drop(late_ibtracs[late_ibtracs].index)

        # For rows without IBTrACS, take the start of ERA5 track
        late_era5 = table[table.id_ibtracs == ""].storm_start.dt.year > end_year
        table = table.drop(late_era5[early_era5].index)

    return table


def categories(
    table: pd.DataFrame,
    subset: str,
    invests: bool = False,
    label: str = "era5",
) -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """Separate a matching summary into verification categories.

    Rows selected by ``subset`` are evaluated against their IBTrACS matches. An
    observed cyclone is counted as a hit if any row with its ``id_ibtracs`` value is
    selected. All rows for observed cyclones with no selected match are returned as
    misses. Selected reanalysis tracks without an IBTrACS match are false alarms,
    unless the same reanalysis track already occurs among the hits.

    Parameters
    ----------
    table
        WCSI matching summary. It must contain ``id_ibtracs``, ``id_era5``, and the
        Boolean column named by ``subset``. When ``invests`` is true, it must also
        contain ``id_superbt_{label}``.
    subset
        Name of the Boolean column defining which rows are selected by the
        classification or filter being evaluated.
    invests
        If true, separate false alarms matched to a SuperBT invest into a fourth
        output table.
    label
        Dataset suffix used to construct the SuperBT identifier column name
        ``id_superbt_{label}``. This is used only when ``invests`` is true.

    Returns
    -------
    hits : pandas.DataFrame
        Selected rows with a non-empty IBTrACS identifier.
    misses : pandas.DataFrame
        Rows belonging to observed cyclones for which no row was selected.
    false_alarms : pandas.DataFrame
        Selected, unmatched reanalysis rows that are not associated with a hit. If
        ``invests`` is true, this excludes rows with a SuperBT invest match.
    invest_matches : pandas.DataFrame, optional
        False alarms with a non-empty ``id_superbt_{label}`` value. Returned only when
        ``invests`` is true.

    Notes
    -----
    Empty strings denote missing IBTrACS and SuperBT identifiers. The returned tables
    preserve the source index and may contain multiple rows for the same observed
    cyclone when the input contains multiple candidate matches.
    """
    hits = table[(table.id_ibtracs != "") & table[subset]]

    misses = table[
        (table.id_ibtracs != "") & ~np.isin(table.id_ibtracs, hits.id_ibtracs)
    ]

    false_alarms = table[
        (table.id_ibtracs == "") & table[subset] & ~np.isin(table.id_era5, hits.id_era5)
    ]

    if invests:
        invests = false_alarms[false_alarms[f"id_superbt_{label}"] != ""]
        false_alarms = false_alarms[false_alarms[f"id_superbt_{label}"] == ""]

        return hits, misses, false_alarms, invests
    else:
        return hits, misses, false_alarms
