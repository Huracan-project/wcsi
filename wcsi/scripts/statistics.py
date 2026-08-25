"""
Calculate summary statistics for different subsets of tracks

Statistics
    - Hits (weak hits), Misses, False Alarms (invests)
    - Probability of Detection, False alarm rate, critical success index
        - Add uncertainty from binomial distribution
"""

import huracanpy
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import xarray as xr

from .. import filters


def main():
    summary = pd.read_parquet("WCSI_summary_all.parquet")
    summary = summary.rename(columns=dict(weak_match="weak_match_era5"))

    stats = _main(summary, start_year=1979, invests=False)
    stats.to_csv("WCSI_scores_1979-2024.csv")
    stats = _main(summary, start_year=2007, invests=True)
    stats.to_csv("WCSI_scores_invests_2007-2024.csv")

    ibtracs = xr.concat(
        [
            huracanpy.load("IBTrACS_1940-2024_dropped.nc"),
            huracanpy.load("IBTrACS_6h_1940-2024_Tropical-Storms.nc"),
        ],
        dim="record",
    )
    era5 = huracanpy.load("ERA5_all_nature.nc").hrcn.sel_id(
        np.unique(summary.loc[summary.WCSI, "id_era5"])
    )
    era5_basin = era5.hrcn.get_basin(convention="ibtracs").values
    era5_track_id = era5.track_id.values
    del era5

    jra3q = huracanpy.load("JRA3Q_nolat-nwc-tcident_WCSI_nature.nc")
    jra3q_basin = jra3q.hrcn.get_basin(convention="ibtracs").values
    jra3q_track_id = jra3q.track_id.values
    del jra3q

    syclops = huracanpy.load(
        "SyCLoPS_classified_ERA5_1940_2024_6hr.parquet", rename=dict(tid="track_id")
    ).hrcn.sel_id(np.unique(summary.id_syclops))
    syclops_basin = syclops.hrcn.get_basin(convention="ibtracs").values
    syclops_track_id = syclops.track_id.values
    del syclops

    for basin in ["NA", "WP", "EP", "NI", "SI", "SP"]:
        print(basin)
        ibtracs_ids = np.unique(ibtracs.track_id[ibtracs.basin == basin])
        era5_ids = np.unique(era5_track_id[era5_basin == basin])
        jra3q_ids = np.unique(jra3q_track_id[jra3q_basin == basin])
        syclops_ids = np.unique(syclops_track_id[syclops_basin == basin])

        notibtracs = summary.id_ibtracs == ""
        summary_ = summary[
            np.isin(summary.id_ibtracs, ibtracs_ids)
            | (np.isin(summary.id_era5, era5_ids) & notibtracs)
            | (np.isin(summary.id_jra3q, jra3q_ids) & notibtracs)
            | (np.isin(summary.id_syclops, syclops_ids) & notibtracs)
        ]
        stats = _main(summary_, start_year=1979, invests=False, subsets=["WCSI"])
        stats.to_csv(f"WCSI_scores_{basin}_1979-2024.csv")
        stats = _main(summary_, start_year=2007, invests=True, subsets=["WCSI"])
        stats.to_csv(f"WCSI_scores_{basin}_invests_2007-2024.csv")


def _main(
    summary,
    start_year,
    invests=False,
    subsets=["all", "H2017-nolat", "H2017", "WCS", "WCSI"],
):
    summary = filters.year(summary, start_year)

    tracks = ["era5", "jra3q", "syclops"]

    total = summary.id_ibtracs.nunique() - 1

    stats = []
    for name in tracks:
        print(name)
        table = summary[(summary.id_ibtracs != "") | (summary[f"id_{name}"] != -1)]
        if name == "era5":
            for subset in subsets:
                print(subset)

                if subset == "all":
                    table_ = table
                else:
                    table_ = table[table[subset]]

                row = get_stats(table_, total, name, invests=invests)
                row["subset"] = f"era5_{subset}"
                stats.append(row)
        else:
            row = get_stats(table, total, name, invests=invests)
            row["subset"] = name
            stats.append(row)

    return pd.DataFrame(stats)


def get_stats(table, total, label, invests=False):

    matches = table[
        (table[f"id_{label}"] != -1)
        & ~table[f"weak_match_{label}"]
        & (table.id_ibtracs != "")
    ]
    weak_matches = table[table[f"weak_match_{label}"]]
    weak_matches = weak_matches[~np.isin(weak_matches.id_ibtracs, matches.id_ibtracs)]

    # Total numbers
    hit = matches.id_ibtracs.nunique()
    weak_hit = weak_matches.id_ibtracs.nunique()
    miss = total - hit - weak_hit
    fa = table.loc[
        (table[f"id_{label}"] != -1) & (table.id_ibtracs == ""), f"id_{label}"
    ].nunique()

    stats = dict(
        hit=hit,
        weak_hit=weak_hit,
        miss=miss,
        false_alarm=fa,
    )

    # Scores
    statistic_with_errors(stats, "pod", hit, total)
    statistic_with_errors(stats, "pod_weak", hit + weak_hit, total)
    statistic_with_errors(stats, "far", fa, hit + fa)
    statistic_with_errors(stats, "csi", hit, total + fa)

    if invests:
        invest = table.loc[
            (table[f"id_superbt_{label}"] != "") & (table.id_ibtracs == ""),
            f"id_{label}",
        ].nunique()
        stats["invest"] = invest
        statistic_with_errors(stats, "far_invest", fa - invest, hit + fa - invest)
        statistic_with_errors(stats, "csi_invest", hit, total + fa - invest)

    return stats


def statistic_with_errors(stats, label, x, y):
    ci = binomtest(x, y).proportion_ci()
    stats[label] = x / y
    stats[f"{label}_low"] = ci.low
    stats[f"{label}_high"] = ci.high


if __name__ == "__main__":
    main()
