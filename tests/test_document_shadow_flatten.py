import pytest
import imaging.document_shadow_flatten as mod

CASES = [
    ("shadow_lowpass_separate", (1,)),
    ("kernel_by_resolution", (1,)),
    ("text_region_protect", (1,)),
    ("gain_per_region", (1,)),
    ("batch_baseline", (1,)),
    ("edge_feather", (1,)),
    ("load", (1,)),
    ("flatten", (1,)),
    ("binarize", (1,)),
    ("export", (1,)),
    ("validate_shadow", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
