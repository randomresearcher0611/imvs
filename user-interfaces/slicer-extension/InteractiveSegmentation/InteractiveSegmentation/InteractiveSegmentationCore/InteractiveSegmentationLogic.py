import logging

import slicer
from slicer.ScriptedLoadableModule import *
from numpy import stack as np_stack
import numpy as np

from InteractiveSegmentationLib.MaskRefinerAPIClient import MaskRefinerAPIClient
from InteractiveSegmentationLib.InteractiveModelTrainingAPIClient import InteractiveModelTrainingAPIClient


class InteractiveSegmentationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

        self.successfully_tested_api_urls = set([])
        self.volume_arr = None
        self.segment_arr = None

        self.maskRefinerApiClient = MaskRefinerAPIClient("http://localhost:5000")
        self.interactiveModelTrainingAPIClient = InteractiveModelTrainingAPIClient("http://localhost:5001")

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
    
    def load_volume_and_segmentation(self):
        self.volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        self.volume_arr = slicer.util.arrayFromVolume(self.volumeNode)

        segment_name = "Segment"
        self.segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        segmentation = self.segmentationNode.GetSegmentation()
        self.segmentId = segmentation.GetSegmentIdBySegmentName(segment_name)
        self.segment_arr = slicer.util.arrayFromSegmentBinaryLabelmap(self.segmentationNode, self.segmentId, self.volumeNode)

    def set_volume_and_segmentation(self):
        if self.volume_arr is not None:
            print("    ---> volume update not programmed")

        if self.segment_arr is not None:
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                self.segment_arr,
                self.segmentationNode,
                self.segmentId,
                self.volumeNode
            )

    def get_current_slice_index(self, slice_type):
        slice_index = -1

        if slice_type == "axial":
            sliceNodeID = "vtkMRMLSliceNodeRed"
            sliceNode = slicer.mrmlScene.GetNodeByID(sliceNodeID)
            appLogic = slicer.app.applicationLogic()
            sliceLogic = appLogic.GetSliceLogic(sliceNode)
            sliceOffset = sliceLogic.GetSliceOffset()
            slice_index = sliceLogic.GetSliceIndexFromOffset(sliceOffset) - 1
        
        return slice_index
    
    def test_server_status(self, api_url):
        if api_url in self.successfully_tested_api_urls:
            return True
        
        self.successfully_tested_api_urls.add(api_url)
        return True
    
    def validate_config(self, config):
        errors = []

        if config['interactive_model_training']['enabled']:
            if not self.test_server_status(config['interactive_model_training']['api_url']):
                pass
        
        if config['mask_tracker']['enabled']:
            if not self.test_server_status(config['mask_tracker']['api_url']):
                pass
        
        if config['mask_refiner']['enabled']:
            if not self.test_server_status(config['mask_refiner']['api_url']):
                pass
        
        return errors
    
    def construct_pipeline_from_config(self, config):
        pipeline = []

        current_slice_index = 0
        num_axial_slices = 100

        tracking_start_index = 0
        tracking_end_index = num_axial_slices
        
        ## Tracker: apply slice limits
        if config['mask_tracker']['config']['slice_limits']['min'] != -1:
            tracking_start_index = max(0, config['mask_tracker']['config']['slice_limits']['min'])
        if config['mask_tracker']['config']['slice_limits']['max'] != -1:
            tracking_end_index = min(num_axial_slices, config['mask_tracker']['config']['slice_limits']['max'])
        
        ## Tracker: check forward and backward ends
        if config['mask_tracker']['config']['tracking_mode'] == "tracking_until_end":
            tracking_start_index = current_slice_index
        elif config['mask_tracker']['config']['tracking_mode'] == "tracking_until_both_ends":
            pass # No changes
        elif config['mask_tracker']['config']['tracking_mode'] == "track_until_custom_limit":
            tracking_start_index = max(
                tracking_start_index,
                current_slice_index - config['mask_tracker']['config']['custom_tracking_limit']['prev']
            )
            tracking_end_index = min(
                tracking_end_index,
                current_slice_index - config['mask_tracker']['config']['custom_tracking_limit']['next']
            )


        # IML Stage
        if config['interactive_model_training']['enabled']:
            pipeline.append(('interactive_model_training', current_slice_index, current_slice_index))
        
        # Refine IML Output before tracking
        if config['mask_refiner']['enabled']:
            if config['mask_refiner']['config']['refine_current_slice_mask_before_tracking']:
                pipeline.append(('mask_refiner', current_slice_index, current_slice_index))
        
        if config['mask_tracker']['enabled']:
            # Track forwards: current slice -> greater
            pipeline.append(('mask_tracker', current_slice_index, tracking_end_index))
            
            # Track backwards: current slice -> lesser
            pipeline.append(('mask_tracker', current_slice_index, tracking_start_index))
        
        # Refine all updated masks
        if config['mask_refiner']['enabled']:
            pipeline.append(('mask_refiner', tracking_start_index, tracking_end_index))
        
        return pipeline
    
    def segment_current_slice(self):
        print("")
        print("Executing pipeline:")

        self.load_volume_and_segmentation()
        print("  ---> volume and segmentation load ok")

        current_axial_slice_index = self.get_current_slice_index("axial")

        current_slice_arr = self.segment_arr[current_axial_slice_index]
        updated_current_slice_arr = self.interactiveModelTrainingAPIClient.segment_single_slice(current_slice_arr, current_axial_slice_index)
        print(np.unique(updated_current_slice_arr))
        self.segment_arr[current_axial_slice_index] = updated_current_slice_arr

        self.set_volume_and_segmentation()
        print("  ---> volume and segmentation set ok")
    
    def interactive_segment_current_slice(self):
        self.load_volume_and_segmentation()

        slice_index = self.get_current_slice_index("axial")
        slice_np_arr = self.volume_arr[slice_index]
        slice_segmentation_np_arr = self.segment_arr[slice_index]
        refined_slice_segmentation_np_arr = self.interactiveModelTrainingAPIClient.interactive_segment_single_slice(
            slice_np_arr,
            slice_segmentation_np_arr,
            slice_index
        )
        self.segment_arr[slice_index] = refined_slice_segmentation_np_arr
        
        self.set_volume_and_segmentation()
        print("  ---> volume and segmentation set ok")

    def refine_current_slice(self):
        self.load_volume_and_segmentation()

        slice_index = self.get_current_slice_index("axial")

        slice_np_arr = self.volume_arr[slice_index]
        slice_np_arr_rgb = np_stack((slice_np_arr,) *3 , axis=-1)
        slice_segmentation_np_arr = self.segment_arr[slice_index]
        refined_slice_segmentation_np_arr = self.maskRefinerApiClient.refine_mask(
            slice_np_arr_rgb,
            slice_segmentation_np_arr,
            slice_index
        )
        self.segment_arr[slice_index] = refined_slice_segmentation_np_arr

        self.set_volume_and_segmentation()
        print("  ---> volume and segmentation set ok")


    def update_segmentation_by_executing_pipeline(self, pipeline):
        print("")
        print("Executing pipeline:")

        self.load_volume_and_segmentation()
        print("  ---> volume and segmentation load ok")

        for step in pipeline:
            if step[0] == "interactive_model_training":
                slice_index = step[1]

                slice_np_arr = self.volume_arr[slice_index]
                slice_segmentation_np_arr = self.segment_arr[slice_index]
                refined_slice_segmentation_np_arr = self.interactiveModelTrainingAPIClient.interactive_segment_single_slice(
                    slice_np_arr,
                    slice_segmentation_np_arr,
                    slice_index
                )
                self.segment_arr[slice_index] = refined_slice_segmentation_np_arr
            elif step[0] == "mask_tracker":
                pass
            elif step[0] == "mask_refiner":
                start_slice_index = step[1]
                end_slice_index = step[2]
                
                signum = 1 if start_slice_index == end_slice_index else \
                    round((end_slice_index - start_slice_index) / abs(end_slice_index - start_slice_index))
                for slice_index in range(start_slice_index, end_slice_index + signum, signum):
                    slice_np_arr = self.volume_arr[slice_index]
                    slice_np_arr_rgb = np_stack((slice_np_arr,) *3 , axis=-1)
                    slice_segmentation_np_arr = self.segment_arr[slice_index]
                    refined_slice_segmentation_np_arr = self.maskRefinerApiClient.refine_mask(
                        slice_np_arr_rgb,
                        slice_segmentation_np_arr,
                        slice_index
                    )
                    self.segment_arr[slice_index] = refined_slice_segmentation_np_arr
        
        self.set_volume_and_segmentation()
        print("  ---> volume and segmentation set ok")

    def test(self):
        current_axial_slice_index = self.get_current_slice_index("axial")
        pipeline = [
            ("interactive_model_training", current_axial_slice_index, current_axial_slice_index)
            # ("mask_refiner", current_axial_slice_index, current_axial_slice_index)
        ]

        print("Executing pipeline:")
        print(pipeline)

        self.update_segmentation_by_executing_pipeline(pipeline)
