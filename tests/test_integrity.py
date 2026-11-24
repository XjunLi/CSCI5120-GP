"""Regression tests for data leakage, cohort identity and measured budgets."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from golfbench.io import digest
from golfbench.space import make_space,make_manifest,validate_manifest
from golfbench.table import build_table,TableOracle
from golfbench.confirmation import confirmation_plan
from golfbench.search import run_search
from golfbench.analysis import summarize_runs


def fixture(profile="cpu_smoke"):
    manifest=make_manifest(make_space(profile),"CPU" if profile=="cpu_smoke" else "H200")
    records=[]
    for job in manifest["jobs"]:
        records.append({**{k:job[k] for k in ["candidate_id","config","training_seed","backend","split","max_steps","train_budget_seconds","device_class"]},
                        "status":"completed","val_bpb":4.1,"quantized_bpb":4.2,
                        "artifact_bytes":10000,"train_seconds":1.0 if profile=="cpu_smoke" else 601.0,
                        "end_to_end_seconds":2.0 if profile=="cpu_smoke" else 610.0,
                        "evaluation_fingerprint":"search-fixture","hardware_fingerprint":"test-hardware",
                        "execution_fingerprint":"test-execution","provenance":{"source":"unit_test_fixture"}})
    return manifest,records


class IntegrityTests(unittest.TestCase):
    def test_h200_default_is_72_ten_minute_jobs(self):
        manifest=make_manifest(make_space())
        self.assertEqual(manifest["device_class"],"H200")
        self.assertEqual(len(manifest["jobs"]),72)
        self.assertEqual(manifest["training_budget_gpu_hours"],12)
        self.assertEqual({j["train_budget_seconds"] for j in manifest["jobs"]},{600})
        self.assertFalse(manifest["long_calibration_enabled"])
        validate_manifest(manifest)

    def test_missing_and_duplicate_runs_are_not_imputed(self):
        manifest,records=fixture()
        with self.assertRaisesRegex(ValueError,"incomplete"):
            build_table(manifest,records[:-1],"smoke")
        with self.assertRaisesRegex(ValueError,"duplicate"):
            build_table(manifest,records+[records[0]],"smoke")

    def test_changed_code_and_hardware_do_not_mix(self):
        for field in ["execution_fingerprint","hardware_fingerprint","evaluation_fingerprint"]:
            manifest,records=fixture()
            records[0][field]="changed"
            with self.assertRaisesRegex(ValueError,"mixed"):
                build_table(manifest,records,"smoke")

    def test_confirmation_record_cannot_enter_search_table(self):
        manifest,records=fixture()
        records[0]["split"]="confirm"
        with self.assertRaises(ValueError):
            build_table(manifest,records,"smoke")

    def test_gpu_approximate_time_checked_independently(self):
        manifest,records=fixture("gpu")
        table=build_table(manifest,records,"short")
        self.assertTrue(all(r["feasible"] for r in table["rows"]))
        records[0]["train_seconds"]=1000
        records[0]["end_to_end_seconds"]=1010
        records[0]["budget_compliant"]=True
        table=build_table(manifest,records,"short")
        self.assertFalse(next(r for r in table["rows"] if r["candidate_id"]==records[0]["candidate_id"])["feasible"])

    def test_manifest_and_table_tampering_detected(self):
        manifest,records=fixture()
        table=build_table(manifest,records,"smoke")
        table["rows"][0]["val_bpb"]=0.001
        with self.assertRaisesRegex(ValueError,"checksum"):
            TableOracle(table)
        manifest["jobs"][0]["config"]["num_layers"]=20
        with self.assertRaisesRegex(ValueError,"identity"):
            validate_manifest(manifest)

    def test_fresh_seeds_and_immutable_winner_required(self):
        manifest,records=fixture()
        table=build_table(manifest,records,"smoke")
        oracle=TableOracle(table)
        result=run_search(oracle.candidates(),oracle.evaluate,"random",0)
        lock={"schema_version":1,"table_id":table["table_id"],"experiment_id":"fixture",
              "training_seeds_used_for_search":[0,1,2],
              "winners":[{"method":result["method"],"search_seed":result["search_seed"],"winner":result["winner"],"winner_lock":result["winner_lock"]}]}
        lock["lock_id"]=digest(lock)
        plan=confirmation_plan([lock],table,[100,101,102])
        self.assertEqual(len(plan["jobs"]),3)
        with self.assertRaisesRegex(ValueError,"fresh"):
            confirmation_plan([lock],table,[0])
        altered=copy.deepcopy(lock)
        altered["winners"][0]["winner"]["config"]["num_layers"]=999
        altered["lock_id"]=digest({k:v for k,v in altered.items() if k!="lock_id"})
        with self.assertRaisesRegex(ValueError,"config"):
            confirmation_plan([altered],table,[100])

    def test_aggregate_uses_search_seed_and_reports_failed_pairs(self):
        manifest,records=fixture()
        table=build_table(manifest,records,"smoke")
        runs=[]
        for method in ["harness-adaptive","harness-frozen"]:
            oracle=TableOracle(table)
            runs.append(run_search(oracle.candidates(),oracle.evaluate,method,1))
        runs[0]["winner"]=None
        summary=summarize_runs(runs,table)
        pair=summary["paired_comparisons"][0]
        self.assertEqual(pair["frozen_only_feasible_pairs"],1)
        self.assertEqual(pair["both_feasible_pairs"],0)
        self.assertIsNone(pair["mean_adaptive_minus_frozen_bpb"])


if __name__=="__main__":
    unittest.main()
