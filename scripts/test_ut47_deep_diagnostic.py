"""CPU controls for the diagnostic supervisor, not a substitute for eight-GPU acceptance."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

FILE=Path(__file__).with_name("ut47_deep_diagnostic.py")
if not FILE.exists(): FILE=Path(__file__).parents[1]/"scripts/ut47_deep_diagnostic.py"
spec=importlib.util.spec_from_file_location("deep",FILE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Controls(unittest.TestCase):
    def test_graph_identity(self):
        import hashlib
        g=m.graph_data(47)
        semantic={k:g[k] for k in ("schema","action","start")}
        self.assertEqual(hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
                         "ff71582e6018295a7b1d23605f350623df9059f4e3d595c25ec29a75f9666e7e")
    def test_cpu_oracle(self):
        self.assertEqual(sum(m.oracle(3)),729)
        self.assertEqual(m.oracle(3)[0],1)
    def test_no_measured_fallback(self):
        r=dict(reports=[dict(status="COMPLETE",tuning=dict(measured=False))])
        self.assertEqual(m.production_verdict(r),"AUTOTUNE_NOT_MEASURED")
    def test_measured_rejected_trial_is_not_selected_profile(self):
        r=dict(reports=[dict(status="COMPLETE",tuning=dict(measured=False,pilots=[dict(measured=True)]))])
        self.assertEqual(m.production_verdict(r),"AUTOTUNE_NOT_MEASURED")
    def test_candidate_failure_not_pass(self):
        r=dict(reports=[dict(status="COMPLETE",tuning=dict(measured=True,candidate_failures=[{"error":"CUDA999"}]))])
        self.assertEqual(m.production_verdict(r),"AUTOTUNE_CANDIDATE_FAILURE")
    def test_resource_not_complete(self):
        r=dict(reports=[dict(status="INCOMPLETE",layers=m.EXPECTED,tuning=dict(global_profile=dict(measured=True)))])
        self.assertEqual(m.production_verdict(r),"AUTOTUNE_AND_BOUNDED_BFS_VERIFIED")
    def test_error_status_not_accepted(self):
        r=dict(reports=[dict(status="ERROR",layers=m.EXPECTED,tuning=dict(measured=True))])
        self.assertEqual(m.production_verdict(r),"UNEXPECTED_BFS_STATUS")
    def test_false_complete_rejected(self):
        r=dict(reports=[dict(status="COMPLETE",layers=m.EXPECTED,tuning=dict(measured=True))])
        self.assertEqual(m.production_verdict(r),"COMPLETE_CARDINALITY_MISMATCH")
    def test_group_contract_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            events=[dict(api="ncclSendContract",group=2,rank=0,count=123,datatype=4)]
            events += [dict(api="ncclCommGetAsyncError",group=2,datatype=7)]*20
            (root/"api-1.ndjson").write_text("\n".join(json.dumps(e) for e in events)+"\n")
            detail=m.failure_details(root)
            self.assertEqual(detail["api"][0]["last_group_contract"][0]["count"],123)
    def test_no_recommendation_from_smoke_only(self):
        entries=[dict(name="baseline-oracle",status="PASS",spec=dict(kind="oracle",mode="baseline"))]
        self.assertEqual(m.analyze(entries,False)["recommended"],[])
    def test_matrix_failure_blocks_recommendation(self):
        entries=[]
        for label,kind in [("oracle","oracle"),("collective","nccl"),("fresh-256","fixed"),("persistent-transitions","fixed"),("production","production")]:
            entries.append(dict(name="baseline-"+label,status="FAIL" if label=="fresh-256" else "PASS",
                                spec=dict(mode="baseline",kind=kind),
                                result=dict(reports=[dict(status="COMPLETE",tuning=dict(measured=True))])))
        self.assertEqual(m.analyze(entries,False)["recommended"],[])
    def test_budget_skip(self):
        with tempfile.TemporaryDirectory() as d:
            result=m.run_stage(Path(d),"skip",dict(kind="fixed"),10,time.monotonic()-1)
            self.assertEqual(result["status"],"SKIPPED_BUDGET")
    def test_owned_timeout_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            child=root/"fake.py"
            child.write_text("import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c','import time;time.sleep(300)'])\ntime.sleep(300)\n")
            with patch.object(m,"__file__",str(child)):
                result=m.run_stage(root,"timeout",dict(kind="fake"),1.1,time.monotonic()+15)
            self.assertEqual(result["status"],"TIMEOUT")
            self.assertTrue(result["cleanup_complete"])
            self.assertTrue((root/"timeout/process-snapshot.json").exists())
    def test_archive_excludes_states_jobs_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"out";root.mkdir()
            (root/"states.bin").write_bytes(b"NO")
            (root/"job-1.json").write_text('{"secret":"NO"}')
            (root/"report.json").write_text('{"status":"FAIL"}')
            (root/"secret.log").symlink_to("/etc/passwd")
            archive=m.bundle(root)
            with zipfile.ZipFile(archive) as z:
                names=z.namelist()
                self.assertNotIn("out/job-1.json",names)
                self.assertNotIn("out/states.bin",names)
                self.assertNotIn("out/secret.log",names)
                self.assertIn("out/report.json",names)
    def test_health_failure_gates_every_gpu_trial(self):
        with tempfile.TemporaryDirectory() as d:
            old=os.getcwd();os.chdir(d)
            seen=[]
            try:
                def fake_stage(base,name,spec,*rest):
                    seen.append(spec["kind"])
                    return dict(name=name,status="FAIL",cleanup_complete=True,spec=spec,result={})
                with patch.object(m.subprocess,"run"), patch.object(m,"build_tracer",return_value=None), \
                     patch.object(m,"run_stage",side_effect=fake_stage), patch.object(m,"bundle"):
                    code=m.main(type("Args",(),dict(budget=300))())
                self.assertEqual(code,1)
                self.assertEqual(seen,["health"])
            finally:os.chdir(old)
    def test_latest_phase_selected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            m.write(root/"phase-1.json",dict(rank=0,layer=1,native_call="old",updated_unix=1))
            m.write(root/"rank-0.phase.json",dict(rank=0,layer=4,native_call="latest",updated_unix=2))
            detail=m.failure_details(root)
            self.assertEqual(len(detail["native"]),1)
            self.assertEqual(detail["native"][0]["native_call"],"latest")
            self.assertEqual(detail["phase_file_count"],2)
    def test_prebuilt_tracer_loads(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            result=m.build_tracer(root)
            self.assertIsNotNone(result)
            self.assertEqual(json.loads((root/"tracer.json").read_text())["status"],"PREBUILT_VERIFIED")
    def test_last_native_call_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            m.write(root/"phase-1.json",dict(rank=7,layer=1,native_call="ncclGroupEnd",native_call_state="entered"))
            self.assertEqual(m.failure_details(root)["native"][0]["native_call"],"ncclGroupEnd")

if __name__=="__main__":
    unittest.main(verbosity=2)
