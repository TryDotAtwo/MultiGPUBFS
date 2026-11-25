import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
from multigpubfs import GraphDefinition
from multigpubfs.cli import main
class CliContract(unittest.TestCase):
 def test_json_and_explicit_devices_preserve_contract(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'g.json';g=GraphDefinition.permutation([[1,0]],[-9,1<<40]);p.write_text(g.to_json())
   with patch('multigpubfs.cli.run_graph',return_value={'status':'COMPLETE'}) as run:
    self.assertEqual(main([str(p),str(Path(d)/'out'),'--devices','1,0','--seconds','12','--backend','generic','--no-autotune']),0)
    self.assertEqual(run.call_args.args[0].digest(),g.digest());self.assertEqual(run.call_args.kwargs['devices'],[1,0]);self.assertFalse(run.call_args.kwargs['autotune'])
 def test_memory_backend_passed_to_native_launch(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'g.json';p.write_text(GraphDefinition.permutation([[1,0]],[0,1]).to_json())
   for backend in ('legacy','segmented24','permutation_bitmap'):
    with patch('multigpubfs.cli.run_graph',return_value={'status':'COMPLETE'}) as run:
     self.assertEqual(main([str(p),str(Path(d)/backend),'--memory-backend',backend]),0)
     self.assertEqual(run.call_args.kwargs['memory_backend'],backend)
 def test_segmented_cli_forwards_custom_rows(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'g.json';p.write_text(GraphDefinition.permutation([[1,0]],[0,1]).to_json())
   with patch('multigpubfs.cli.run_graph',return_value={'status':'COMPLETE'}) as run:
    self.assertEqual(main([str(p),str(Path(d)/'out'),'--memory-backend','segmented','--segment-rows','65536']),0)
    self.assertEqual(run.call_args.kwargs['segment_rows'],65536)
    self.assertEqual(run.call_args.kwargs['memory_backend'],'segmented')
 def test_bad_definition_does_not_start_gpu(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'bad.json';p.write_text('{}')
   with patch('multigpubfs.cli.run_graph') as run:self.assertEqual(main([str(p),str(Path(d)/'out')]),1);run.assert_not_called()


def test_window_lineage_preflight_keeps_failed_cli_output_available(tmp_path, monkeypatch):
    import pytest
    from multigpubfs import launch
    from multigpubfs.api import BfsSession

    pytest.importorskip("pyarrow")
    graph = tmp_path / "graph.json"
    graph.write_text(GraphDefinition.permutation([[1, 0]], [0, 1]).to_json(), encoding="utf-8")
    output = tmp_path / "result"
    lineage = tmp_path / "lineage"
    lineage.touch()
    visited = []

    def fake_launch(graph, root, *, _path_window, **options):
        assert _path_window is None
        visited.append(root.parent.parent)
        return dict(status="UNREACHABLE", stop_reason=None, d_min=None,
                    target_index=None, generator_ids=[])

    monkeypatch.setattr(BfsSession, "_native_nearest", lambda *args, **kwargs: ("native", {}, (0, 1), True))
    monkeypatch.setattr(launch, "run_graph", fake_launch)

    def run(out, scratch):
        return main(["run", str(graph), str(out), "--gpus", "2", "--targets", "[[1,0]]",
                     "--goal-mode", "window", "--lineage-dir", str(scratch)])

    assert run(output, lineage) == 1
    assert not output.exists()
    assert visited == []

    lineage.unlink()
    assert run(output, lineage) == 0
    assert (output / "report.json").is_file()

    nested_output = tmp_path / "nested_result"
    nested_lineage = nested_output / "csr"
    assert run(nested_output, nested_lineage) == 0
    assert visited == [lineage.resolve(), nested_lineage.resolve()]
    assert (nested_output / "report.json").is_file()


if __name__=='__main__':unittest.main()
