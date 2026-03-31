"""
Filters for the summary table of WCSI and matching across datasets
"""

import numpy as np


def year(table, start_year):

    # Select by ibtracs year
    early_ibtracs = (
        table[table.id_ibtracs != ""].id_ibtracs.str.slice(0, 4).astype(int)
        < start_year
    )
    table = table.drop(early_ibtracs[early_ibtracs].index)

    # For rows without IBTrACS, take the start of ERA5 track
    early_era5 = table[table.id_ibtracs == ""].storm_start.dt.year < start_year
    table = table.drop(early_era5[early_era5].index)

    return table


def categories(table, subset, invests=False):
    hits = table[(table.id_ibtracs != "") & table[subset] & ~table.weak_match]

    weak_hits = table[(table.id_ibtracs != "") & table[subset] & table.weak_match]

    misses = table[
        (table.id_ibtracs != "")
        & ~table[subset]
        & ~np.isin(table.id_ibtracs, hits.id_ibtracs)
        & ~table.weak_match
    ]

    false_alarms = table[(table.id_ibtracs == "") & table[subset]]

    if invests:
        false_alarm_invests = false_alarms[false_alarms.id_superbt != ""]
        false_alarms = false_alarms[false_alarms.id_superbt == ""]

        return hits, weak_hits, misses, false_alarms, false_alarm_invests
    else:
        return hits, weak_hits, misses, false_alarms
