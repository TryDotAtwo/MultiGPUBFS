import pytest
from multigpubfs.autotune import _segmented_deep_prefix_candidates

def p(label,depth):
    return dict(candidate=label,layer_seconds=[1.0]*depth,layer_sizes=list(range(depth+1)),reason='PROFILE')

def test_deep_common_prefix_excludes_slow_short_candidate():
    chosen,excluded=_segmented_deep_prefix_candidates([p('a',36),p('b',30),p('slow',15)])
    assert [x['candidate'] for x in chosen]==['a','b']
    assert excluded[0]['candidate']=='slow'
    assert excluded[0]['reason']=='SHORT_COMMON_PREFIX_RELATIVE_TO_BEST_COMPLETED_DEPTH'

def test_keep_original_when_only_one_has_deep_prefix():
    chosen,excluded=_segmented_deep_prefix_candidates([p('a',36),p('b',14),p('c',15)])
    assert len(chosen)==3 and not excluded

def test_reject_wrong_prefix_even_if_short():
    bad=p('bad',15);bad['layer_sizes'][9]=999
    with pytest.raises(RuntimeError,match='PREFIX_CORRECTNESS'):
        _segmented_deep_prefix_candidates([p('a',36),p('b',30),bad])
