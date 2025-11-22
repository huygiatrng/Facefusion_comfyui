from typing import Any, Dict, Literal, TypeAlias, List
from numpy.typing import NDArray

InputTypes : TypeAlias = Dict[str, Any]

NodeClassMapping : TypeAlias = Dict[str, Any]
NodeDisplayNameMapping : TypeAlias = Dict[str, str]

FaceSwapperModel = Literal[
    'hyperswap_1a_256',
    'hyperswap_1b_256', 
    'hyperswap_1c_256',
    'inswapper_128',
    'inswapper_128_fp16',
    'blendswap_256',
    'simswap_256',
    'simswap_unofficial_512',
    'uniface_256'
]

# Face detection types
Face : TypeAlias = Dict[str, Any]
FaceData : TypeAlias = Dict[str, Any]
