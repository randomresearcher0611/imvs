from slicer.util import warningDisplay


def tryParseInt(inputString, label, defaultValue):
    if len(inputString.strip()) == 0:
        return defaultValue
    
    try:
        inputInt = int(inputString)
        return inputInt
    except ValueError:
        warningDisplay(
            f"The value at {label} is not a valid integer, "
            f"the default value {defaultValue} will be used."
            "Please leave this input field empty or provide a valid integer value."
        )

        return defaultValue
