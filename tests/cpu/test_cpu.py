import pytest

import scPrisma


def test_cpuonly_import():
    assert scPrisma.spectrum_gen.jit.__name__ == "jit"
    assert scPrisma.algorithms.__name__ =="scPrisma.algorithms"
    with pytest.raises(AttributeError) as exc_info:
        scPrisma.algorithms_torch
    assert (
        "module 'scPrisma' has no attribute 'algorithms_torch'" in exc_info.value.args[0]
    )