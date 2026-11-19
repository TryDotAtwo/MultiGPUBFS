import pytest
from multigpubfs import GraphDefinition
from multigpubfs.launch import run_graph

def test_bitmap_rejects_matrix_before_native_initialization(tmp_path,monkeypatch):
 def fail(*a,**kw):raise AssertionError("native must not run")
 monkeypatch.setattr("multigpubfs.launch.native_runtime",fail)
 g=GraphDefinition.matrix(1,1,[([1],0)],[1])
 with pytest.raises(ValueError,match="BITMAP_REQUIRES_PERMUTATION_ACTION"):
  run_graph(g,tmp_path/"x",memory_backend="permutation_bitmap")

def test_bitmap_rejects_unsupported_degree_before_native_initialization(tmp_path,monkeypatch):
 def fail(*a,**kw):raise AssertionError("native must not run")
 monkeypatch.setattr("multigpubfs.launch.native_runtime",fail)
 g=GraphDefinition.permutation([list(range(65))],list(range(65)))
 with pytest.raises(ValueError,match="BITMAP_UNSUPPORTED_DEGREE"):
  run_graph(g,tmp_path/"x",memory_backend="permutation_bitmap")
