try:
    from .algorithms_torch import *
except ImportError as torch_err:
    print("""WARNING! Torch is not installed! Falling back to numba for cpu-only execution.
For utilizing gpu please install scPrisma like so pip install .[gpu]""")
    from .algorithms import *

from .pre_processing import *
from .spectrum_gen import *
from .data_gen import *
