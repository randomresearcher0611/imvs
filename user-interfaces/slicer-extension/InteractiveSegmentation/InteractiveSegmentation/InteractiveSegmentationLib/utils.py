from io import BytesIO

from numpy import (
    clip as np_clip,
    load as np_load,
    save as np_save,
    zeros as np_zeros,
    where as np_where,
    uint8,
    uint16,
)


def save_np_arr_to_buffer(np_arr):
    data_buffer = BytesIO()
    
    np_save(data_buffer, np_arr)
    return data_buffer


def load_np_arr_from_stream(stream_content):
    buffer = BytesIO(stream_content)
    
    np_arr = np_load(buffer)
    return np_arr


def window_frame(frame, window_level=90, window_width=120):
    lower_bound = window_level - window_width // 2
    upper_bound = window_level + window_width // 2

    clipped_image = np_clip(frame, lower_bound, upper_bound)

    windowed_image = ((clipped_image - lower_bound) / (upper_bound - lower_bound)) * 255
    windowed_image = windowed_image.astype(uint8)

    return windowed_image


def uint16_mask_to_uint8(uint16_mask):
    uint8_mask = np_where(uint16_mask == 1, 255, uint16_mask).astype(uint8)
    return uint8_mask


def uint8_mask_to_uint16(mask):
    uint16_mask = np_where(mask < 127, 0, 255).astype(uint16)
    return uint16_mask
