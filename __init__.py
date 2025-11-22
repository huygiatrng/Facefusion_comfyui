from .install import install

install()

from .facefusion_api import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ =\
[
	'NODE_CLASS_MAPPINGS',
	'NODE_DISPLAY_NAME_MAPPINGS'
]
