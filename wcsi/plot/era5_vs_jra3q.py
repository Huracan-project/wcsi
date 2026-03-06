from pathlib import Path
from string import ascii_lowercase

import huracanpy
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import numpy as np
import pandas as pd
from scipy.stats import linregress
import seaborn as sb
from tqdm import tqdm


def main(summary):
    sets = []

    is_era5 = summary["H2017-nolat"] & summary["WCSI"] & ~summary["weak_match"]
    is_jra3q = summary["WCSI_jra3q"]
    is_ibtracs = summary.id_ibtracs != ""
    # (100, 010, 110, 001, 101, 011, 111)
    # 100 - Only ERA5
    sets.append(len(np.unique(summary[is_era5 & ~is_jra3q & ~is_ibtracs].id_era5)))

    # 010 - Only JRA3Q
    sets.append(len(np.unique(summary[~is_era5 & is_jra3q & ~is_ibtracs].id_jra3q)))

    # 110 - ERA5 and JRA3Q
    sets.append(len(np.unique(summary[is_era5 & is_jra3q & ~is_ibtracs].id_era5)))

    # 001 - Only IBTrACS
    sets.append(len(np.unique(summary[~is_era5 & ~is_jra3q & is_ibtracs].id_ibtracs)))

    # 101 - ERA5 and IBTrACS
    sets.append(len(np.unique(summary[is_era5 & ~is_jra3q & is_ibtracs].id_ibtracs)))

    # 011 - JRA3Q and IBTrACS
    sets.append(len(np.unique(summary[~is_era5 & is_jra3q & is_ibtracs].id_ibtracs)))

    # 111 - In all datasets
    sets.append(len(np.unique(summary[is_era5 & is_jra3q & is_ibtracs].id_ibtracs)))

    print(sum(sets))

    fig, axes = plt.subplot_mosaic(
        """
        12
        13
        """,
        figsize=(8, 5),
    )

    venn3(
        sets,
        set_labels=["ERA5", "JRA3Q", "IBTrACS"],
        set_colors=["C0", "C1", "C2"],
        alpha=0.65,
        ax=axes["1"],
    )

    if not Path("matched_lmi.parquet").exists():
        ibtracs = huracanpy.load("IBTrACS_6h_1940-2024_Tropical-Storms.nc")
        tracks_era5 = huracanpy.load("ERA5_all.nc")
        tracks_jra3q = huracanpy.load("JRA3Q_nolat-tcident.nc")
        matched_lmi = match_lmi(ibtracs, tracks_era5, tracks_jra3q)
        matched_lmi.to_parquet("matched_lmi.parquet")
    else:
        matched_lmi = pd.read_parquet("matched_lmi.parquet")

    x = matched_lmi.mslp_ibtracs
    for dataset, ax in [("era5", axes["2"]), ("jra3q", axes["3"])]:
        y = matched_lmi[f"mslp_{dataset}"]
        sb.kdeplot(x=x, y=y, ax=ax, fill=True)
        result = linregress(x=x[matched_lmi.year < 1979], y=y[matched_lmi.year < 1979])
        ax.plot(
            [870, 1020],
            result.slope * np.array([870, 1020]) + result.intercept,
            "C1",
            label=f"1948-1979 ({result.slope:.3f} $\pm$ {result.stderr:.3f})",
        )
        result = linregress(
            x=x[matched_lmi.year >= 1979], y=y[matched_lmi.year >= 1979]
        )
        ax.plot(
            [870, 1020],
            result.slope * np.array([870, 1020]) + result.intercept,
            "C2",
            label=f"1979-2024 ({result.slope:.3f} $\pm$ {result.stderr:.3f})",
        )
        ax.plot(x, y, ".k", alpha=0.5, ms=0.5)
        ax.plot([870, 1020], [870, 1020], "-k")
        ax.set(xlim=(870, 1020), ylim=(870, 1020))

    axes["2"].set_xticklabels([])
    axes["2"].set(xlabel="", ylabel="MSLP ERA5 (hPa)")
    axes["3"].set(
        xlabel="MSLP minimum per track in IBTrACS (hPa)", ylabel="MSLP JRA3Q (hPa)"
    )
    axes["2"].legend()
    axes["3"].legend()

    axes["1"].set_title("Matching")

    for n, label in enumerate(axes):
        ax = axes[label]
        ax.text(0.05, 1.05, f"({ascii_lowercase[n]})", transform=ax.transAxes)

    fig.suptitle("ERA5 vs JRA3Q")
    plt.savefig("era5_vs_jra3q.pdf")


def match_lmi(ibtracs, tracks_era5, tracks_jra3q):
    ibtracs_tc_lmi = ibtracs.isel(
        record=np.where((ibtracs.nature == "TS") & ~np.isnan(ibtracs.slp))[0]
    ).hrcn.get_apex_vals("slp", stat="min")

    track_ids = ibtracs_tc_lmi.track_id.values
    ibtracs_tc_lmi = ibtracs_tc_lmi.rename(track_id="record").drop_vars("record")
    ibtracs_tc_lmi = ibtracs_tc_lmi.assign(track_id=("record", track_ids))

    matched_lmi = huracanpy.assess.match(
        [ibtracs_tc_lmi, tracks_era5, tracks_jra3q],
        ["ibtracs", "era5", "jra3q"],
        max_dist=165,
    )
    matched_lmi = matched_lmi[matched_lmi.id_ibtracs.astype(str) != "nan"]

    matched_lmi["year"] = np.zeros(len(matched_lmi), dtype=int)
    matched_lmi["lat"] = np.zeros(len(matched_lmi))
    for dataset in ["ibtracs", "era5", "jra3q"]:
        matched_lmi[f"mslp_{dataset}"] = np.zeros(len(matched_lmi))

    for n, row in tqdm(matched_lmi.iterrows()):
        lmi_ib = ibtracs_tc_lmi.hrcn.sel_id(track_id=row.id_ibtracs)
        matched_lmi.loc[n, "year"] = lmi_ib.time.dt.year.values[()]
        matched_lmi.loc[n, "mslp_ibtracs"] = lmi_ib.slp.values[()]
        matched_lmi.loc[n, "lat"] = lmi_ib.lat.values[()]
        for dataset, tracks in [("era5", tracks_era5), ("jra3q", tracks_jra3q)]:
            if ~np.isnan(row[f"id_{dataset}"]):
                track = tracks.hrcn.sel_id(row[f"id_{dataset}"])
                track = track.isel(record=np.where(track.time == lmi_ib.time.values)[0])
                matched_lmi.loc[n, f"mslp_{dataset}"] = track.mslp.values[()]

    matched_lmi = matched_lmi[
        ~np.isnan(matched_lmi.id_era5) & ~np.isnan(matched_lmi.id_jra3q)
    ]
    return matched_lmi


if __name__ == "__main__":
    summary = pd.read_parquet("WCSI_summary_all.parquet")
    summary = summary[summary.storm_start.dt.year >= 1979]
    main(summary)
