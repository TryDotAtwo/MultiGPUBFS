from types import SimpleNamespace
import pytest
from multigpubfs import generic_session as m

def test_group_selection_validates_once_and_binds_generation(monkeypatch):
 calls=[];pool=SimpleNamespace(generation=4,workers=[],select=lambda *a:calls.append(a))
 monkeypatch.setattr(m,'_active',pool)
 selected=m.launch_group_selection('/bin/native',[0,1],{})
 assert len(calls)==1 and selected[1]==4
 pool.generation=5
 with pytest.raises(RuntimeError,match='SELECTION_CHANGED'):m.launch(['/bin/native','graph-rank','g','p','0'],{},None,[0,1],_selection=selected)

def test_group_selection_rejects_dead_worker(monkeypatch):
 worker=SimpleNamespace(process=SimpleNamespace(poll=lambda:1))
 pool=SimpleNamespace(generation=4,workers=[worker],select=lambda *a:None)
 monkeypatch.setattr(m,'_active',pool);selected=m.launch_group_selection('/bin/native',[0],{})
 with pytest.raises(RuntimeError,match='FAILED_GROUP_NO_RESTART'):m.launch(['/bin/native','graph-rank','g','p','0'],{},None,[0],_selection=selected)
