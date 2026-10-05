from datetime import timedelta
from itertools import groupby
from pathlib import Path

from cartopy.crs import EqualEarth, Geodetic, PlateCarree
from cyclopts import App
import huracanpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
import xarray as xr

from wcsi.summary import filters

app = App()

projection = EqualEarth
transform = Geodetic()

extents = dict(
    NATL=[-100, 40, 5, 70],
    MED=[-120, 60, 0, 90],
    ENP=[-180, 0, 0, 90],
    CP=[120, 300, 0, 90],
    WNP=[60, 240, 0, 90],
    NI=[20, 160, 5, 70],
    SI=[20, 160, -65, 0],
    AUS=[60, 240, -90, 0],
    SP=[120, 300, -90, 0],
    SA=[-120, 60, -90, 0],
)

# extents = dict(
#     NATL=[-120, 60, 0, 90],
#     MED=[-120, 60, 0, 90],
#     ENP=[-180, 0, 0, 90],
#     CP=[120, 300, 0, 90],
#     WNP=[60, 240, 0, 90],
#     NI=[0, 180, 0, 90],
#     SI=[0, 180, -90, 0],
#     AUS=[60, 240, -90, 0],
#     SP=[120, 300, -90, 0],
#     SA=[-120, 60, -90, 0],
# )

# Colour for each cyclone category
colorsb = mpl.colormaps["tab20b"](np.linspace(0, 1, 20))
colorsc = mpl.colormaps["tab20c"](np.linspace(0, 1, 20))
natures = dict(
    wcsi=dict(
        TC=("Tropical", colorsc[4]),
        Tr=("Trough", colorsc[8]),
        Ot=("Other", colorsb[12]),
        BC=("Baroclinic", colorsc[0]),
        MV=("Mid-level", colorsc[12]),
        Vo=("Vortex", colorsc[16]),
    ),
    ibtracs=dict(
        TS=("Tropical", colorsc[5]),
        SS=("Subtropical", colorsb[19]),
        NR=("Not Recorded", colorsb[11]),
        ET=("Extratropical", colorsc[1]),
        MX=("Mixture", colorsb[3]),
        DB=("Disturbance", colorsc[17]),
    ),
    superbt=dict(
        TS=("Tropical", colorsc[5]),
        SS=("Subtropical", colorsb[19]),
        LO=("Low", colorsb[3]),
        NT=("Unknown", colorsc[10]),
        EX=("Extratropical", colorsc[2]),
        WV=("Wave", colorsb[7]),
        DB=("Disturbance", colorsc[18]),
    ),
)

# Layout of the figure for plt.subplot_mosaic
mosaic = """
    lAA
    lAA
    lAA
    lAA
    lAA
    lAA
    mAA
    mAA
    mAA
    mAA
    mAA
    mAA
    000
    x1y
    B1C
    B1C
    B1C
    B1C
    D1E
    D1E
    D1E
    D1E
    F1G
    F1G
    F1G
    F1G
"""


def track_overview(
    track=None, obs=None, dataset_label="ERA5", obs_label="IBTrACS", obs_legend_ncol=2
):
    if track is not None:
        tmin, tmax = track.time.min(), track.time.max()
        if obs is not None:
            tmin = min(tmin, obs.time.min())
    elif obs is not None:
        tmin, tmax = obs.time.min(), obs.time.max()
    else:
        raise ValueError("Must pass at least one track")

    basin = guess_basin(track, obs)
    fig, axes = create_figure(basin, tmin, tmax)
    lines_axes = []

    if track is not None:
        # Calculate track parameters
        # Cyclone phase space
        b = uniform_filter1d(np.abs(track.cps_b), size=5, mode="nearest")
        vtl = uniform_filter1d(track.cps_vtl, size=5, mode="nearest")
        vtu = uniform_filter1d(track.cps_vtu, size=5, mode="nearest")

        vo850 = track.relative_vorticity.sel(pressure=850)
        vo200 = track.relative_vorticity.sel(pressure=200)

        vorticity = uniform_filter1d(vo850, size=5, mode="nearest")
        dvort = vo850 - vo200

        mapplot(
            axes["A"],
            track,
            linewidth=5,
            color_mapping=natures["wcsi"],
            ax_legend=axes["l"],
            title=f"{dataset_label} Nature",
            bbox_to_anchor=[0.75, 0.8],
            ncol=2,
        )

    # IBTrACS overlay
    if obs is not None:
        mapplot(
            axes["A"],
            obs,
            linewidth=2,
            color_mapping=natures[obs_label.lower()],
            ax_legend=axes["m"],
            title=f"{obs_label} Nature",
            bbox_to_anchor=[0.75, 0.8],
            ncol=obs_legend_ncol,
        )

    # Fill for different criteria
    if track is not None:
        for ax_label in ["B", "C", "D", "E", "F", "G"]:
            nature_background(
                axes[ax_label],
                track.nature.values,
                pd.to_datetime(track.time),
                natures["wcsi"],
            )

    if obs is not None:
        for ax_label in ["x", "y"]:
            nature_background(
                axes[ax_label],
                obs.nature.values,
                pd.to_datetime(obs.time),
                natures[obs_label.lower()],
            )

    if track is not None:
        # Intensity
        lb0, lb1, lb2 = intensityplot(
            axes["B"],
            axes["D"],
            axes["F"],
            track.time,
            track.mslp,
            vorticity,
            track.vmax10m,
            label=dataset_label,
        )
        lines_axes += [(lb0, axes["B"]), (lb1, axes["D"]), (lb2, axes["F"])]

        # CPS
        lc0, lc1, lc2 = cpsplot(
            axes["C"], axes["E"], axes["G"], track.time, b, vtl, vtu, dvort
        )
        lines_axes += [(lc0, axes["C"]), (lc1, axes["E"]), (lc2, axes["G"])]

        axes["E"].legend(bbox_to_anchor=[0, 0.9])

    if obs is not None:
        lb0, lb1, lb2 = intensityplot(
            axes["B"],
            axes["D"],
            axes["F"],
            obs.time,
            obs.mslp,
            None,
            obs.wind,
            color="C7" if track is not None else "k",
            linestyle="--",
            label="IBTrACS",
        )

        if track is None:
            lines_axes += [(lb0, axes["B"]), (lb2, axes["F"])]

    if track is not None and obs is not None:
        axes["B"].legend(bbox_to_anchor=[1, 1])

    for line, axis in lines_axes:
        axis.yaxis.label.set_color(line[0].get_color())
        axis.tick_params(axis="y", colors=line[0].get_color())

    axes["A"].set_extent(extents[basin], crs=PlateCarree())

    return fig, axes


def guess_basin(track, ib):
    if track is not None:
        basin = track.hrcn.get_basin().values[np.argmax(track.vorticity.values)]
    else:
        try:
            basin = ib.hrcn.get_basin().values[np.argmax(ib.wind.values)]
        except ValueError:
            try:
                basin = ib.hrcn.get_basin().values[np.argmin(ib.mslp.values)]
            except ValueError:
                basin = ib.hrcn.get_basin().values[0]

    return basin


def create_figure(basin, tmin, tmax):
    central_lon = 0.5 * (extents[basin][0] + extents[basin][1])
    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(8, 9),
        per_subplot_kw=dict(
            A=dict(projection=projection(central_longitude=central_lon))
        ),
    )

    axes["l"].set_axis_off()
    axes["m"].set_axis_off()

    axes["0"].remove()
    axes["1"].remove()
    plt.subplots_adjust(wspace=0, hspace=0)

    axes["A"].set_extent(extents[basin], crs=PlateCarree())
    axes["A"].stock_img()
    axes["A"].coastlines()
    axes["A"].gridlines(
        xlocs=range(extents[basin][0], extents[basin][1] + 1, 30),
        ylocs=range(extents[basin][2], extents[basin][3], 15),
        draw_labels=["left", "bottom"],
        color="w",
        zorder=-1,
    )

    axes["B"].set(
        ylim=(950, 1025),
        yticks=[950, 975, 1000, 1025],
        ylabel="MSLP (hPa)",
    )

    axes["D"].set(
        ylim=(0, 50),
        yticks=[0, 6, 25, 50],
        ylabel="$\\xi_\\mathrm{850hPa}$\n(10$^{-5}$ s$^{-1}$)",
    )
    axes["F"].set(
        ylim=(0, 80), yticks=[0, 40, 80], ylabel=r"$v_\mathrm{max, 10m}$ (m s$^{-1}$)"
    )

    axes["C"].set(
        ylim=(0, 120),
        yticks=[0, 15, 60, 120],
        ylabel="Asymmetry (B)",
    )
    axes["E"].set(
        ylim=(-300, 300),
        yticks=[-300, 0, 300],
        ylabel=r"Thermal Wind (V$_\mathrm{T}$)",
    )
    axes["G"].set(
        ylim=(-20, 20),
        yticks=[-20, 0, 6, 20],
        ylabel=(
            "$\\xi_\\mathrm{850hPa} - \\xi_\\mathrm{200hPa}$\n(10$^{-5}$ s$^{-1}$)"
        ),
    )

    axes["x"].set(ylim=(0, 1), title="Intensity")
    axes["y"].set(ylim=(0, 1), title="Structure")

    for ax_label in ["B", "C", "D", "E", "F", "G", "x", "y"]:
        axes[ax_label].set(xlim=(tmin, tmax), clip_on=False)

        for direction in ["top", "right"]:
            axes[ax_label].spines[direction].set_visible(False)

    for ax_label in ["x", "y", "B", "C", "D", "E"]:
        axes[ax_label].spines["bottom"].set_visible(False)
        axes[ax_label].get_xaxis().set_ticks([])

    for ax_label in ["D", "E"]:
        axes[ax_label].yaxis.tick_right()
        axes[ax_label].yaxis.set_label_position("right")
        axes[ax_label].spines["left"].set_visible(False)
        axes[ax_label].spines["right"].set_visible(True)

    for ax_label in ["x", "y"]:
        axes[ax_label].spines["left"].set_visible(False)
        axes[ax_label].get_yaxis().set_ticks([])

    fig.autofmt_xdate()

    axes["A"].text(0, 1.05, "(a)", transform=axes["A"].transAxes)
    axes["x"].text(0, 1.05, "(b)", transform=axes["x"].transAxes)
    axes["y"].text(0, 1.05, "(c)", transform=axes["y"].transAxes)

    return fig, axes


def mapplot(ax, track, linewidth, color_mapping, ax_legend=None, **kwargs):
    track.hrcn.plot_fancyline(colors="w", ax=ax, linewidths=linewidth + 1)

    idx = 0
    for nat_, npoints in [
        (a[0], sum([1 for _ in a[1]])) for a in groupby(track.nature.values)
    ]:
        track.isel(
            record=slice(idx, min(idx + npoints + 1, len(track.time)))
        ).hrcn.plot_fancyline(
            color=color_mapping[nat_][1],
            ax=ax,
            linewidths=linewidth,
            alphas=1,
        )
        idx += npoints

    if ax_legend is not None:
        for label in color_mapping:
            ax_legend.fill_betweenx(
                [np.nan, np.nan],
                np.nan,
                np.nan,
                color=color_mapping[label][1],
                label=color_mapping[label][0],
            )
        return ax_legend.legend(**kwargs, columnspacing=0.5)


def intensityplot(
    ax, axt1, axt2, time, mslp, vorticity, wind, color="k", linestyle="-", label="ERA5"
):
    # Intensity
    l0 = ax.plot(time, mslp, color, linestyle=linestyle, label=label)

    if vorticity is not None:
        l1 = axt1.plot(time, vorticity, color, linestyle=linestyle)
        showlimit(axt1, time, vorticity, 6, color=color)
        vorticity_masked = np.where(np.gradient(vorticity) <= 0, np.nan, vorticity)
        axt1.fill_between(time, vorticity_masked, 0, color=color, alpha=0.25)
        axt1.axhline(0, color=color)
    else:
        l1 = None

    l2 = axt2.plot(time, wind, color, linestyle=linestyle)

    return l0, l1, l2


def cpsplot(ax, axt1, axt2, time, b, vtl, vtu, dvort):
    l0 = ax.plot(time, b, "-k")
    showlimit(ax, time, b, 15, above=False, color="k")

    l1 = axt1.plot(time, vtl, "-C3", label=r"V$_\mathrm{T}^\mathrm{L}$")
    l2 = axt1.plot(time, vtu, "--C3", label=r"V$_\mathrm{T}^\mathrm{U}$")
    showlimit(axt1, time, vtl, 0, color="C3")
    showlimit(axt1, time, vtu, 0, color="C3")

    l3 = axt2.plot(time, dvort, "-k")
    showlimit(axt2, time, dvort, 6, color="k")

    return l0, l1, l3


def nature_background(ax, nature, times, color_mapping):
    y = ax.get_ylim()

    idx = 0
    for nat_, npoints in [(a[0], sum([1 for _ in a[1]])) for a in groupby(nature)]:
        ax.fill_betweenx(
            [y[0], y[1]],
            times[idx] - timedelta(hours=3),
            times[idx + npoints - 1] + timedelta(hours=3),
            color=color_mapping[nat_][1],
            ec=None,
            alpha=0.6,
        )

        idx += npoints


def combined_legend(ax, lines):
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)


def showlimit(ax, x, y, ylim, above=True, color="k"):
    ax.axhline(ylim, color=color)

    if above:
        y_masked = np.where(y >= ylim, y, np.nan)
    else:
        y_masked = np.where(y < ylim, y, np.nan)

    ax.fill_between(x, y_masked, ylim, color=color, alpha=0.25)


@app.default
def main(
    data_path: Path = Path("."),
    reanalysis_fname: str = "ERA5.nc",
    ibtracs_fname: list[str] | None = None,
    superbt_fname: str = "superbt.nc",
    summary_fname: str = "WCSI_summary_all.parquet",
    reanalysis_label="era5",
    plot_path: Path = Path("."),
):
    """
    Parameters
    ----------
    data_path
    era5_fname
    ibtracs_fname
    superbt_fname
    summary_fname
    plot_path
    """
    tracks = huracanpy.load(str(data_path / reanalysis_fname))

    if ibtracs_fname is None:
        ibtracs_fname = [
            "IBTrACS_6h_1940-2024_Tropical-Storms.nc",
            "IBTrACS_1940-2024_dropped.nc",
        ]
    ibtracs = xr.concat(
        [huracanpy.load(str(data_path / fname)) for fname in ibtracs_fname],
        dim="record",
    )
    superbt = huracanpy.load(str(data_path / superbt_fname))
    # Simplify superBT natures
    superbt.nature[np.isin(superbt.nature, ["TD", "HU", "TY", "TC", "ST"])] = "TS"
    superbt.nature[superbt.nature == "SD"] = "SS"
    superbt.nature[superbt.nature == "MD"] = "DB"
    superbt.nature[superbt.nature == "PT"] = "EX"
    superbt.nature[superbt.nature == "TW"] = "WV"

    table = pd.read_parquet(str(data_path / summary_fname))
    if reanalysis_label != "era5":
        table["WCSI"] = table[f"id_{reanalysis_label}"] != -1

    # Specific tracks for WCSI paper
    _main(tracks, ibtracs, superbt, table, reanalysis_label, plot_path)

    # Plot all tracks
    plot_all(tracks, ibtracs, superbt, table, reanalysis_label, plot_path)


def _main(tracks, ibtracs, superbt, table, reanalysis_label, plot_path):
    # Only show subset of natures that appear in tracks for paper
    natures_ibtracs = natures["ibtracs"]
    natures["ibtracs"] = {key: natures_ibtracs[key] for key in ["TS", "ET"]}
    natures_superbt = natures["superbt"]
    natures["superbt"] = {key: natures_superbt[key] for key in ["LO", "DB"]}

    # True positive Iris
    table_ = table[table.id_ibtracs == "1995235N13311"]
    track = tracks.hrcn.sel_id(table_[f"id_{reanalysis_label}"])
    ib = ibtracs.hrcn.sel_id(table_.id_ibtracs)
    fig, ax = track_overview(
        track, ib, dataset_label=reanalysis_label.upper(), obs_legend_ncol=1
    )
    ax["A"].set_title("Hurricane Iris 1995")
    fig.savefig(plot_path / "iris_1995.pdf")
    plt.close(fig)

    # False negative Earl
    table_ = table[table.id_ibtracs == "1986254N22309"]
    track = tracks.hrcn.sel_id(table_[f"id_{reanalysis_label}"])
    ib = ibtracs.hrcn.sel_id(table_.id_ibtracs)
    fig, ax = track_overview(
        track, ib, dataset_label=reanalysis_label.upper(), obs_legend_ncol=1
    )
    ax["A"].set_title("Hurricane Earl 1986")
    fig.savefig(plot_path / "earl_1986.pdf")
    plt.close(fig)

    if reanalysis_label == "era5":
        # Invest
        table_ = table[table[f"id_superbt_{reanalysis_label}"] == "b8l.2013"]
        track = tracks.hrcn.sel_id(table_[f"id_{reanalysis_label}"])
        ib = superbt.hrcn.sel_id(table_[f"id_superbt_{reanalysis_label}"])
        fig, ax = track_overview(
            track,
            ib,
            dataset_label=reanalysis_label.upper(),
            obs_label="SuperBT",
            obs_legend_ncol=1,
        )
        ax["A"].set_title("Invest b8l 2013")
        fig.savefig(plot_path / "invest_b8l_2013.pdf")
        plt.close(fig)

        # False positives
        table_ = table[np.isin(table.id_era5, [67491, 67214, 195394])]
        titles = [
            "Monsoon Low",
            "Polar Low",
            "Weak Tropical Storm",
        ]
        for m, (n, row) in enumerate(tqdm(table_.iterrows())):
            track = tracks.hrcn.sel_id(row.id_era5)
            fig, ax = track_overview(track)
            ax["A"].set_title(f"False Positive {titles[m]} ({row.storm_start.year})")
            fig.savefig(
                plot_path / f"false-positive_{row.id_era5}_{row.storm_start.year}.pdf"
            )
            plt.close(fig)

    natures["ibtracs"] = natures_ibtracs
    natures["superbt"] = natures_superbt


def plot_all(tracks, ibtracs, superbt, table, reanalysis_label, plot_path):
    hits, weak_hits, misses, false_alarms, invests = filters.categories(
        table, "WCSI", invests=True, label=reanalysis_label
    )

    (plot_path / "hit").mkdir(exist_ok=True)
    for prefix, subset in [("hit", hits), ("weak_hit", weak_hits)]:
        for n, row in tqdm(subset.iterrows(), total=len(subset)):
            track_id = row[f"id_{reanalysis_label}"]
            track = tracks.hrcn.sel_id(track_id)
            ib = ibtracs.hrcn.sel_id(row.id_ibtracs)
            fig, axes = track_overview(
                track=track, obs=ib, dataset_label=reanalysis_label.upper()
            )
            fig.savefig(plot_path / "hit" / f"{prefix}_{row.id_ibtracs}_{track_id}.png")
            plt.close(fig)

    (plot_path / "miss").mkdir(exist_ok=True)
    for n, row in tqdm(misses.iterrows(), total=len(misses)):
        ib = ibtracs.hrcn.sel_id(row.id_ibtracs)
        if row[f"id_{reanalysis_label}"] == -1:
            track = None
            track_id = ""
        else:
            track_id = row[f"id_{reanalysis_label}"]
            track = tracks.hrcn.sel_id(row[track_id])
        fig, axes = track_overview(
            track=track, obs=ib, dataset_label=reanalysis_label.upper()
        )
        fig.savefig(plot_path / "miss" / f"miss_{row.id_ibtracs}_{track_id}.png")
        plt.close(fig)

    (plot_path / "invest").mkdir(exist_ok=True)
    for n, row in tqdm(invests.iterrows(), total=len(invests)):
        track_id = row[f"id_{reanalysis_label}"]
        track = tracks.hrcn.sel_id(row[f"id_{reanalysis_label}"])
        ib = superbt.hrcn.sel_id(row[f"id_superbt_{reanalysis_label}"])
        fig, axes = track_overview(
            track=track,
            obs=ib,
            obs_label="SuperBT",
            dataset_label=reanalysis_label.upper(),
        )
        fig.savefig(
            plot_path
            / "invest"
            / f"invest_{row[f'id_superbt_{reanalysis_label}']}_{track_id}.png"
        )
        plt.close(fig)

    (plot_path / "false_alarm").mkdir(exist_ok=True)
    for n, row in tqdm(false_alarms.iterrows(), total=len(false_alarms)):
        track_id = row[f"id_{reanalysis_label}"]
        track = tracks.hrcn.sel_id(track_id)
        fig, axes = track_overview(track=track, dataset_label=reanalysis_label.upper())
        fig.savefig(plot_path / "false_alarm" / f"false_alarm_{track_id}.png")
        plt.close(fig)


if __name__ == "__main__":
    app()
