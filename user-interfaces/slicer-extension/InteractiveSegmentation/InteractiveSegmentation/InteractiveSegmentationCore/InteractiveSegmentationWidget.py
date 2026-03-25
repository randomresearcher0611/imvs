import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

from InteractiveSegmentationCore.InteractiveSegmentationLogic import InteractiveSegmentationLogic
from InteractiveSegmentationCore.utils import tryParseInt


class InteractiveSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/InteractiveSegmentation.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = InteractiveSegmentationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(  slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        ###

        # Buttons
        self.ui.runSegmentorButtonPushButton.connect("clicked(bool)", self.segment_current_slice)
        # self.ui.test1PushButton.connect("clicked(bool)", self.test_logic)
        self.ui.runImlPipelinePushButton.connect("clicked(bool)", self.handleRunImlPipelineClick)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    #TODO: to implement the below method
    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        ###

        # Update buttons states and tooltips
        ###

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    #TODO: to implement the below method
    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def getConfigFromGUI(self):
        tracking_mode = (
            "track_until_end" if self.ui.maskTrackingTrackingModeTrackUntilEndRadioButton.isChecked() else
            "track_until_both_ends" if self.ui.maskTrackingTrackingModeTrackUntilBothEndsRadioButton.isChecked() else
            "track_until_custom_limit" if self.ui.maskTrackingTrackingModeTrackPrevNextRadioButton.isChecked() else
            None
        )

        config = {
            'interactive_model_training': {
                'enabled': self.ui.interactiveModelTrainingEnabledCheckBox.checked,
                'api_url': self.ui.interactiveModelTrainingApiUrlLineEdit.text,
            },
            'mask_tracker': {
                'enabled': self.ui.maskTrackingEnabledCheckBox.checked,
                'api_url': self.ui.maskTrackingApiUrlLineEdit.text,
                'config': {
                    'slice_limits': {
                        'min': tryParseInt(
                            self.ui.maskTrackingSliceMinLimitLineEdit.text,
                            "Mask Tracker -> Apply Slice Limits -> Min Slice Number",
                            -1
                        ),
                        'max': tryParseInt(
                            self.ui.maskTrackingSliceMaxLimitLineEdit.text,
                            "Mask Tracker -> Apply Slice Limits -> Max Slice Number",
                            -1
                        ),
                    },
                    'tracking_mode': tracking_mode,
                    'custom_tracking_limit': {
                        'prev': tryParseInt(
                            self.ui.maskTrackingTrackingModeTrackPrevNextPrevSlicesLineEdit.text,
                            "Mask Tracker -> Tracking Mode -> Track Prev-Next -> Num Prev Slices",
                            -1
                        ),
                        'next': tryParseInt(
                            self.ui.maskTrackingTrackingModeTrackPrevNextNextSlicesLineEdit.text,
                            "Mask Tracker -> Tracking Mode -> Track Prev-Next -> Num Next Slices",
                            -1
                        ),
                    },
                },
            },
            'mask_refiner': {
                'enabled': self.ui.maskRefinerEnabledCheckBox.checked,
                'api_url': self.ui.maskRefinerApiUrlLineEdit.text,
                'config': {
                    'refine_current_slice_mask_before_tracking': self.ui.maskRefinerRefineCurrentBeforeTrackingCheckBox.checked,
                },
            },
        }

        print("Config from GUI")
        print(config)
        print("")
        
        return config

    def handleRunImlPipelineClick(self):
        # config = self.getConfigFromGUI()
        
        # config_errors = self.logic.validate_config(config)
        # if len(config_errors) > 0:
        #     pass

        # pipeline = self.logic.construct_pipeline_from_config(config)
        # self.logic.update_segmentation_by_executing_pipeline(pipeline)

        if self.ui.interactiveModelTrainingEnabledCheckBox.checked:
            self.logic.interactive_segment_current_slice()
        
        if self.ui.maskRefinerEnabledCheckBox.checked:
            self.logic.refine_current_slice()

        
    
    def test_logic(self):
        self.logic.test()
    
    def segment_current_slice(self):
        self.logic.segment_current_slice()
