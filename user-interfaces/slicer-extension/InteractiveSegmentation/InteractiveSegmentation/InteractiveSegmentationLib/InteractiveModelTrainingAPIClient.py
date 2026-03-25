from requests import Session
from numpy import stack as np_stack
import cv2

foo_dir = "E:/iml-debug"

from InteractiveSegmentationLib.utils import (
    load_np_arr_from_stream,
    save_np_arr_to_buffer,
    window_frame,
    uint16_mask_to_uint8,
    uint8_mask_to_uint16,
)
import numpy as np


class InteractiveModelTrainingAPIClient:
    def __init__(self, base_url):
        self.requests_session = Session()
        self.model_inference_endpoint = base_url + "/api/segment-single-frame"
        self.goo = base_url + "/api/interactive-segment-single-frame"
    
    def segment_single_slice(self, slice_np_arr, slice_index):
        request_data_buffer = save_np_arr_to_buffer(slice_np_arr)

        response = self.requests_session.post(
            self.model_inference_endpoint,
            data=request_data_buffer.getvalue()
        )
        
        if response.status_code != 200:
            print(f"Failed to refine mask on slice index {slice_index}")
            return np.zeros_like(slice_np_arr)
        
        refined_masks_arr = load_np_arr_from_stream(response.content)

        print(np.unique(refined_masks_arr, return_counts=True))

        # refined_slice_segmentation_np_arr = uint8_mask_to_uint16(refined_masks_arr[0])
        return refined_masks_arr
    
    def interactive_segment_single_slice(self, slice_np_arr, slice_segmentation, slice_index):
        foo = np.stack([slice_np_arr, slice_segmentation], axis=0)

        request_data_buffer = save_np_arr_to_buffer(foo)

        response = self.requests_session.post(
            self.goo,
            data=request_data_buffer.getvalue()
        )
        
        if response.status_code != 200:
            print(f"Failed to refine mask on slice index {slice_index}")
            return slice_segmentation
        
        refined_masks_arr = load_np_arr_from_stream(response.content)

        print(np.unique(refined_masks_arr, return_counts=True))

        # refined_slice_segmentation_np_arr = uint8_mask_to_uint16(refined_masks_arr[0])
        return refined_masks_arr
    
