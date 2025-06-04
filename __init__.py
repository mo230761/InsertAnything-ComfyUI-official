from .insert_anything_node import MaskOption, ReduxProcess, FillProcess, CropBack
from .no_scaling import FillProcessNoScaling, CropBackNoScaling

NODE_CLASS_MAPPINGS = {
    "MaskOption": MaskOption,
    "ReduxProcess": ReduxProcess,
    "FillProcess": FillProcess,
    "CropBack": CropBack,
    "FillProcessNoScaling": FillProcessNoScaling,
    "CropBackNoScaling": CropBackNoScaling,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskOption": "InsertAnything - Mask Option",
    "ReduxProcess": "InsertAnything - Redux Process",
    "FillProcess": "InsertAnything - Fill Process",
    "CropBack": "InsertAnything - Crop Back",
    "FillProcessNoScaling": "InsertAnything - Fill Process (No Scaling)",
    "CropBackNoScaling": "InsertAnything - Crop Back (No Scaling)",
}
