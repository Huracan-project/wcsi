"""
Filters for the summary table of WCSI and matching across datasets
"""

import numpy as np


def year(table, start_year, end_year=None):

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


def categories(table, subset, invests=False, label="era5"):
    hits = table[
        (table.id_ibtracs != "") & table[subset] & ~table[f"weak_match_{label}"]
    ]

    weak_hits = table[
        (table.id_ibtracs != "") & table[subset] & table[f"weak_match_{label}"]
    ]

    misses = table[
        (table.id_ibtracs != "")
        & ~table[subset]
        & ~np.isin(table.id_ibtracs, hits.id_ibtracs)
        & ~table[f"weak_match_{label}"]
    ]

    false_alarms = table[(table.id_ibtracs == "") & table[subset]]

    if invests:
        invests = false_alarms[false_alarms[f"id_superbt_{label}"] != ""]
        false_alarms = false_alarms[false_alarms[f"id_superbt_{label}"] == ""]

        return hits, weak_hits, misses, false_alarms, invests
    else:
        return hits, weak_hits, misses, false_alarms
