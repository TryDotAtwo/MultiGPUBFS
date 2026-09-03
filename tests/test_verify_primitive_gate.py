import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location("verifier",Path(__file__).parents[1]/"scripts/verify_primitive_gate.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def fixture():
    return dict(status="PASS_PRIMITIVE_GATE",source_commit="a"*40,
        gpus=[dict(index=0,name="Tesla T4",uuid="GPU-a"),dict(index=1,name="Tesla T4",uuid="GPU-b")],
        results=[dict(gpu=gpu,test=test,tool=tool,status="PASS")
            for gpu in ["GPU-a","GPU-b"]
            for test in ["generate","hash","route","owner","pipeline","materialize","dense_device","ping_pong"]
            for tool in ["plain","memcheck","racecheck","initcheck","synccheck"]])

class VerifyGateTests(unittest.TestCase):
    def test_query_ctest_inventory_is_explicit(self):
        old = "1/1 Test #1: allocation-query ... Passed 0.00 sec\n100% tests passed, 0 tests failed out of 1"
        v.verify_queries(old, False)
        with self.assertRaises(ValueError): v.verify_queries(old, True)
        new = old.replace("out of 1", "out of 2") + "\n2/2 Test #2: route-query ... Passed 0.10 sec"
        v.verify_queries(new, True)
        with self.assertRaises(ValueError): v.verify_queries(new.replace("route-query", "other"), True)

    def test_pipeline_inventory_cannot_pass_with_only_one_successful_test(self):
        names = [
            "failure_with_both_slots_in_flight_is_sticky_and_drains_on_drop",
            "generation_variants_small_feedback",
            "reused_slots_and_partial_tails_preserve_every_layer",
        ]
        log = "\n".join(f"test {name} ... ok" for name in names)
        v.verify_inventory(log, "ping_pong", "racecheck")
        with self.assertRaises(ValueError):
            v.verify_inventory(log.splitlines()[0], "ping_pong", "racecheck")
        with self.assertRaises(ValueError):
            v.verify_inventory(log, "ping_pong", "plain")
        with self.assertRaises(ValueError):
            v.verify_inventory(log.replace(names[1], "unexpected_test"), "ping_pong", "racecheck")

    def test_full_matrix_requires_unique_results_for_two_distinct_t4_devices(self):
        s=fixture()
        entries=v.verify_summary(s,"a"*40)
        self.assertEqual(len(entries),80)
        self.assertIn(("gpu1-generate-racecheck.log","racecheck"),entries)
        for change in range(6):
            bad=fixture()
            if change==0: bad["results"].pop()
            elif change==1: bad["results"][-1]=bad["results"][0]
            elif change==2: bad["gpus"][1]["uuid"]="GPU-a"
            elif change==3: bad["status"]="INCOMPLETE"
            elif change==4: bad["source_commit"]="b"*40
            else: bad["gpus"][1]["name"]="RTX 3070"
            with self.assertRaises(ValueError): v.verify_summary(bad,"a"*40)

    def test_passing_rust_tests_without_sanitizer_exit_summary_are_incomplete(self):
        rust="test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 186.14s\n"
        v.verify_log(rust,"plain")
        with self.assertRaises(ValueError): v.verify_log(rust,"racecheck")
        v.verify_log(rust+"========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n","racecheck")
        v.verify_log(rust+"========= ERROR SUMMARY: 0 errors\n","memcheck")
        with self.assertRaises(ValueError): v.verify_log(rust+"========= ERROR SUMMARY: 1 errors\n","memcheck")
        with self.assertRaises(ValueError): v.verify_log(rust.replace("4 passed","0 passed"),"plain")

if __name__=="__main__": unittest.main()
