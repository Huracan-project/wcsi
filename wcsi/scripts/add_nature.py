import huracanpy
import numpy as np
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

from .. import nature


def main():
    tracks = huracanpy.load("ERA5_all.nc")

    tracks["nature"] = ("record", np.zeros(len(tracks.record), dtype="U2"))
    for track_id, track in tqdm(tracks.groupby("track_id")):
        # Cyclone phase space
        b = uniform_filter1d(np.abs(track.cps_b), size=5, mode="nearest")
        vtl = uniform_filter1d(track.cps_vtl, size=5, mode="nearest")
        vtu = uniform_filter1d(track.cps_vtu, size=5, mode="nearest")
        vorticity = uniform_filter1d(
            track.relative_vorticity.sel(pressure=850), size=5, mode="nearest"
        )

        nat = nature.nature(b, vtl, vtu, vorticity, track.is_tc.values)

        tracks.nature[tracks.track_id == track_id] = nat

    tracks.hrcn.save("ERA5_all_nature.nc")


if __name__ == "__main__":
    main()
