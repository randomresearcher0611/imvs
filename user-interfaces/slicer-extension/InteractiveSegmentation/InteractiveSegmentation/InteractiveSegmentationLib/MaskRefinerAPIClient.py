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

class MaskRefinerAPIClient:
    def __init__(self, base_url):
        self.requests_session = Session()
        self.model_inference_endpoint = base_url + "/api/models/cascade-psp/infer"
    
    def refine_mask(self, slice_np_arr, slice_segmentation_np_arr, slice_index=-1):
        image_arr = window_frame(slice_np_arr)
        mask_arr = uint16_mask_to_uint8(slice_segmentation_np_arr)

        cv2.imwrite(foo_dir + "/image.png", image_arr)
        cv2.imwrite(foo_dir + "/mask.png", mask_arr)

        # print(image_arr.shape)
        # print(mask_arr.shape)

        # return slice_segmentation_np_arr


        single_input_arr = np_stack(
                [
                    image_arr[:, :, 0],
                    image_arr[:, :, 1],
                    image_arr[:, :, 2],
                    mask_arr,
                ],
                axis=-1
            )
        inputs_np_arr = np_stack([single_input_arr], axis=0)
        
        request_data_buffer = save_np_arr_to_buffer(inputs_np_arr)

        response = self.requests_session.post(
            self.model_inference_endpoint,
            data=request_data_buffer.getvalue()
        )
        
        if response.status_code != 200:
            print(f"Failed to refine mask on slice index {slice_index}")
            return slice_segmentation_np_arr
        
        refined_masks_arr = load_np_arr_from_stream(response.content)

        refined_slice_segmentation_np_arr = uint8_mask_to_uint16(refined_masks_arr[0])
        return refined_slice_segmentation_np_arr
