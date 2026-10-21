"""Operations on contiguous ragged arrays

Track datasets commonly store observations from many tracks in flat, one-dimensional
arrays. The corresponding ``track_id`` array identifies which track contains each
observation, allowing tracks to have different numbers of points without padding the
entire dataset to a rectangular shape. This module provides operations that respect
the boundaries between those tracks.

The functions assume a *contiguous ragged* representation: every observation for one
track occurs in a single uninterrupted block, and observations within that block are
already in the required order. They do not sort points, verify temporal spacing, or
combine non-contiguous blocks that share a track identifier.

"""

__all__ = ["mask_short", "padded_index", "uniform_filter1d"]


from ._ragged import mask_short, padded_index, uniform_filter1d
