cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(InteractiveSegmentation)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/InteractiveSegmentation")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Anonymized (Anonymized)")
set(EXTENSION_DESCRIPTION "-")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/InteractiveSegmentation.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/InteractiveSegmentation/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(InteractiveSegmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
