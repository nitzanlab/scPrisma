import pytest

import scPrisma


def test_gpu_import():
    assert scPrisma.algorithms_torch.torch.__version__ >= "1.13.0"
    assert scPrisma.spectrum_gen.jit.__name__ == "jit"
    with pytest.raises(AttributeError) as exc_info:
        scPrisma.algorithms
    assert (
        "module 'scPrisma' has no attribute 'algorithms'" in exc_info.value.args[0]
    )
