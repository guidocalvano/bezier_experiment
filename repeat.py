import numpy as np


def single_non_zero_column_index(one_hot_encoded_rows):
    # insert cuda kernel here
    pass


def repeat(items, repeat_counts):
    offsets = np.cumsum(repeat_counts)
    indices = np.arange(
        offsets[-1]
    )

    index_comes_after_offset = indices[:, np.newaxis] < offsets[np.newaxis, :]
    item_per_index = items[single_non_zero_column_index(
            index_comes_after_offset != np.roll(index_comes_after_offset, 1, axis=1)
        )
    ]

    return item_per_index

