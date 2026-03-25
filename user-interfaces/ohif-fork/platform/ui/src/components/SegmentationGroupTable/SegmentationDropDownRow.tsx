import React, { useState } from 'react';
import { Select, Icon, Dropdown, Tooltip } from '../../components';
import PropTypes from 'prop-types';
import { useTranslation } from 'react-i18next';
import {
  Types,
  Enums,
  getEnabledElementByViewportId,
  VolumeViewport,
  utilities,
  StackViewport,
  imageLoader,
} from '@cornerstonejs/core';
import { utilities as csUtils } from '@cornerstonejs/core';
import { eventTarget, EVENTS } from '@cornerstonejs/core';
import {
  Enums as csToolsEnums,
  segmentation as cstSegmentation,
  Types as cstTypes,
  utilities as cstUtils,
} from '@cornerstonejs/tools';

import App, { commandsManager, extensionManager, servicesManager } from '../../../../app/src/App';
import { act } from 'react-dom/test-utils';

function SegmentationDropDownRow({
  segmentations = [],
  activeSegmentation,
  onActiveSegmentationChange,
  disableEditing = false,
  onToggleSegmentationVisibility,
  onSegmentationEdit,
  onSegmentationDownload,
  onSegmentationDownloadRTSS,
  storeSegmentation,
  onSegmentationDelete,
  onSegmentationAdd,
  addSegmentationClassName,
}) {
  const handleChange = option => {
    onActiveSegmentationChange(option.value); // Notify the parent
  };

  const [loading, setLoading] = useState(false);
  // const [loading2, setLoading2] = useState(false);

  const handleClick = async () => {
    setLoading(true);

    const { segmentationService, viewportGridService } = servicesManager.services;
    const { activeViewportId } = viewportGridService.getState();
    const { viewport } = getEnabledElementByViewportId(activeViewportId) || {};
    const imageId = viewport.getCurrentImageId();
    const sliceID = viewport.getCurrentImageIdIndex();

    const segmentationId = activeSegmentation.id;

    // Continue with segmentation logic...
    const labelmapVolume = segmentationService.getLabelmapVolume(segmentationId);
    const { dimensions } = labelmapVolume;
    const scalarData = labelmapVolume.getScalarData();
    const width = dimensions[0];
    const height = dimensions[1];
    const startIndex = sliceID * width * height;
    const slicedData = scalarData.slice(startIndex, startIndex + width * height);

    console.log(`imageID: ${imageId}  sliceID: ${sliceID}`);
    console.log(`Seg_ID: ${segmentationId}`);

    try {
      const image = await imageLoader.loadAndCacheImage(imageId);
      const pixelData = image.getPixelData(); // Uint16Array or Int16Array

      // Convert pixelData to a regular array for transmission
      const pixelDataArray = Array.from(pixelData);
      const old_seg_data_array = Array.from(slicedData);

      // Send pixelData and imageId in the POST request
      const response = await fetch('http://localhost:8501/api/segment-image-interactive', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pixelData: pixelDataArray, old_seg_data: old_seg_data_array }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok: ' + response.statusText);
      }

      const { segmentation_data } = await response.json();
      console.log('Segmentation Data:', segmentation_data);

      // Check if segmentation_data matches the expected size
      if (segmentation_data.length !== width * height) {
        console.error(
          `Segmentation data size (${segmentation_data.length}) does not match labelmap dimensions (${width * height})`
        );
        return;
      }

      // Update the scalar data with the new segmentation data
      for (let i = 0; i < segmentation_data.length; i++) {
        scalarData[startIndex + i] = segmentation_data[i];
      }

      const updatedSegmentSlice = scalarData.slice(startIndex, startIndex + width * height);
      console.log('Updated Segment Slice:', updatedSegmentSlice);

      // Notify about the segmentation data modification
      const eventDetail = { segmentationId: segmentationId };
      const event = new CustomEvent(csToolsEnums.Events.SEGMENTATION_DATA_MODIFIED, {
        detail: eventDetail,
      });
      eventTarget.dispatchEvent(event);

      const segmentation = segmentationService.getSegmentation(segmentationId);

      if (segmentation === undefined) {
        console.error('Segmentation not found!');
        return;
      }

      segmentationService._broadcastEvent(segmentationService.EVENTS.SEGMENTATION_DATA_MODIFIED, {
        segmentation,
      });
    } catch (error) {
      console.error('Error fetching segmentation data:', error);
    } finally {
      setLoading(false); // Reset loading state
    }
  };

  const handleClick2 = async () => {
    setLoading(true);

    const { segmentationService, viewportGridService } = servicesManager.services;
    const { activeViewportId } = viewportGridService.getState();
    const { viewport } = getEnabledElementByViewportId(activeViewportId) || {};
    const imageId = viewport.getCurrentImageId();
    const sliceID = viewport.getCurrentImageIdIndex();

    const segmentationId = activeSegmentation.id;

    console.log(`imageID: ${imageId}  sliceID: ${sliceID}`);
    console.log(`Seg_ID: ${segmentationId}`);

    try {
      const image = await imageLoader.loadAndCacheImage(imageId);
      console.log(image);

      const pixelData = image.getPixelData(); // Uint16Array or Int16Array

      // Convert pixelData to a regular array for transmission
      const pixelDataArray = Array.from(pixelData);

      console.log(pixelData);

      console.log(pixelDataArray.length);
      console.log(pixelDataArray[0]?.length);
      console.log(pixelDataArray[0][0]?.length);

      // Send pixelData and imageId in the POST request
      const response = await fetch('http://localhost:8501/api/segment-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pixelData: pixelDataArray }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok: ' + response.statusText);
      }

      const { segmentation_data } = await response.json();
      console.log('Segmentation Data:', segmentation_data);

      // Continue with segmentation logic...
      const labelmapVolume = segmentationService.getLabelmapVolume(segmentationId);
      const { dimensions } = labelmapVolume;
      const scalarData = labelmapVolume.getScalarData();
      const width = dimensions[0];
      const height = dimensions[1];

      // Check if segmentation_data matches the expected size
      if (segmentation_data.length !== width * height) {
        console.error(
          `Segmentation data size (${segmentation_data.length}) does not match labelmap dimensions (${width * height})`
        );
        return;
      }

      const startIndex = sliceID * width * height;

      // Update the scalar data with the new segmentation data
      for (let i = 0; i < segmentation_data.length; i++) {
        scalarData[startIndex + i] = segmentation_data[i];
      }

      const updatedSegmentSlice = scalarData.slice(startIndex, startIndex + width * height);
      console.log('Updated Segment Slice:', updatedSegmentSlice);

      // Notify about the segmentation data modification
      const eventDetail = { segmentationId: segmentationId };
      const event = new CustomEvent(csToolsEnums.Events.SEGMENTATION_DATA_MODIFIED, {
        detail: eventDetail,
      });
      eventTarget.dispatchEvent(event);

      const segmentation = segmentationService.getSegmentation(segmentationId);

      if (segmentation === undefined) {
        console.error('Segmentation not found!');
        return;
      }

      segmentationService._broadcastEvent(segmentationService.EVENTS.SEGMENTATION_DATA_MODIFIED, {
        segmentation,
      });
    } catch (error) {
      console.error('Error fetching segmentation data:', error);
    } finally {
      setLoading(false); // Reset loading state
    }
  };

  const selectOptions = segmentations.map(s => ({
    value: s.id,
    label: s.label,
  }));
  const { t } = useTranslation('SegmentationTable');

  if (!activeSegmentation) {
    return null;
  }

  return (
    <div className="group mx-0.5 mt-[8px] flex items-center pb-[10px]">
      <div
        onClick={e => {
          e.stopPropagation();
        }}
      >
        <Dropdown
          id="segmentation-dropdown"
          showDropdownIcon={false}
          alignment="left"
          itemsClassName={`text-primary-active ${addSegmentationClassName}`}
          showBorders={false}
          maxCharactersPerLine={30}
          list={[
            ...(!disableEditing
              ? [
                  {
                    title: t('Add new segmentation'),
                    onClick: () => {
                      onSegmentationAdd();
                    },
                  },
                ]
              : []),
            ...(!disableEditing
              ? [
                  {
                    title: t('Rename'),
                    onClick: () => {
                      onSegmentationEdit(activeSegmentation.id);
                    },
                  },
                ]
              : []),
            {
              title: t('Delete'),
              onClick: () => {
                onSegmentationDelete(activeSegmentation.id);
              },
            },
            ...(!disableEditing
              ? [
                  {
                    title: t('Export DICOM SEG'),
                    onClick: () => {
                      storeSegmentation(activeSegmentation.id);
                    },
                  },
                ]
              : []),
            ...[
              {
                title: t('Download DICOM SEG'),
                onClick: () => {
                  onSegmentationDownload(activeSegmentation.id);
                },
              },
              {
                title: t('Download DICOM RTSTRUCT'),
                onClick: () => {
                  onSegmentationDownloadRTSS(activeSegmentation.id);
                },
              },
            ],
          ]}
        >
          <div className="hover:bg-secondary-dark grid h-[28px] w-[28px] cursor-pointer place-items-center rounded-[4px]">
            <Icon name="icon-more-menu"></Icon>
          </div>
        </Dropdown>
      </div>
      {selectOptions?.length && (
        <Select
          id="segmentation-select"
          isClearable={false}
          onChange={handleChange}
          components={{
            DropdownIndicator: () => (
              <Icon
                name="chevron-down-new"
                className="mr-2"
              />
            ),
          }}
          isSearchable={false}
          options={selectOptions}
          value={selectOptions.find(o => o.value === activeSegmentation?.id) || null}
          className="text-aqua-pale h-[26px] w-1/2 text-[13px]"
        />
      )}
      <div className="flex items-center">
        <Tooltip
          position="bottom-right"
          content={
            <div className="flex flex-col">
              <div className="text-[13px] text-white">Series:</div>
              <div className="text-aqua-pale text-[13px]">{activeSegmentation.description}</div>
            </div>
          }
        >
          <Icon
            name="info-action"
            className="text-primary-active"
          />
        </Tooltip>
        <div
          className="hover:bg-secondary-dark mr-1 grid h-[28px] w-[28px] cursor-pointer place-items-center rounded-[4px]"
          onClick={() => onToggleSegmentationVisibility(activeSegmentation.id)}
        >
          {activeSegmentation.isVisible ? (
            <Icon
              name="row-shown"
              className="text-primary-active"
            />
          ) : (
            <Icon
              name="row-hidden"
              className="text-primary-active"
            />
          )}
        </div>

        {/* New Button 2222 */}
        <div
          className={`h-[10px] w-[14px] hover:cursor-pointer ${
            loading ? 'pointer-events-none opacity-50' : 'hover:opacity-60'
          } mr-2 ml-2`}
          onClick={e => {
            e.stopPropagation();
            if (!loading) {
              handleClick2(); // Call the action handler
            }
          }}
        >
          <p style={{ color: 'white' }}>S</p>
        </div>

        {/* New Button */}
        <div
          className={`h-[10px] w-[14px] hover:cursor-pointer ${
            loading ? 'pointer-events-none opacity-50' : 'hover:opacity-60'
          }`}
          onClick={e => {
            e.stopPropagation();
            if (!loading) {
              handleClick(); // Call the new action handler
            }
          }}
        >
          <p style={{ color: 'white' }}>IS</p>
        </div>
      </div>
    </div>
  );
}

SegmentationDropDownRow.propTypes = {
  segmentations: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
    })
  ).isRequired,
  activeSegmentation: PropTypes.shape({
    id: PropTypes.string.isRequired,
    isVisible: PropTypes.bool.isRequired,
  }),
  onActiveSegmentationChange: PropTypes.func.isRequired,
  disableEditing: PropTypes.bool,
  onToggleSegmentationVisibility: PropTypes.func,
  onSegmentationEdit: PropTypes.func,
  onSegmentationDownload: PropTypes.func,
  onSegmentationDownloadRTSS: PropTypes.func,
  storeSegmentation: PropTypes.func,
  onSegmentationDelete: PropTypes.func,
  onSegmentationAdd: PropTypes.func,
};

export default SegmentationDropDownRow;
