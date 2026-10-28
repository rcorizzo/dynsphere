import pytest


def test_aggregate_refuses_incomplete_trial_matrix(tmp_path):
    from research.explanation_replication.aggregate import aggregate
    doc=dict(manifest_id='id',trial_ids=['c|m|37'],methods=['m'],cases=[dict(case_id='c')],seeds=[37])
    with pytest.raises(ValueError,match='terminal'):aggregate(tmp_path,doc)


def test_metric_reverification_allows_only_numeric_roundoff():
    from research.explanation_replication.aggregate import same_metrics
    assert same_metrics({'x':[.1,None],'q95':5},{'x':[.1+1e-15,None],'q95':5})
    assert not same_metrics({'x':[.1,None]},{'x':[.2,None]})
