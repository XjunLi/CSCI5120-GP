"""Analytically check search metrics using labelled software-only fixtures."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from golfbench.analysis import anytime_regret, factorial_effects, FACTORIAL_CONDITIONS


class AnytimeTests(unittest.TestCase):
    def test_query_prefix_uses_post_warm_start_incumbents(self):
        run = {"initial_candidate_ids": ["a", "b", "c", "d"], "budget_attempts": 6,
               "budget_charge_seconds": None, "trajectory": [
                   {"attempt": 4, "best_val_bpb": 6}, {"attempt": 5, "best_val_bpb": 5},
                   {"attempt": 6, "best_val_bpb": 4}]}
        self.assertEqual(anytime_regret(run, 3), 1.5)

    def test_cost_area_cannot_use_a_result_before_paid_completion(self):
        run = {"initial_candidate_ids": ["a", "b", "c", "d"], "budget_charge_seconds": 10,
               "trajectory": [{"attempt": 4, "best_val_bpb": 6, "cumulative_charge_seconds": 4},
                              {"attempt": 5, "best_val_bpb": 5, "cumulative_charge_seconds": 6},
                              {"attempt": 6, "best_val_bpb": 4, "cumulative_charge_seconds": 9}]}
        self.assertAlmostEqual(anytime_regret(run, 3), 13 / 6)
        changed = deepcopy(run); changed['trajectory'][-1]['best_val_bpb'] = 3
        self.assertAlmostEqual(anytime_regret(run, 3) - anytime_regret(changed, 3), 1 / 6)

    def test_missing_incumbent_is_not_imputed(self):
        self.assertIsNone(anytime_regret({}, 3))
        run = {"initial_candidate_ids": ["a"], "budget_attempts": 2,
               "trajectory": [{"attempt": 1, "best_val_bpb": None}, {"attempt": 2, "best_val_bpb": None}]}
        self.assertIsNone(anytime_regret(run, 3))

    def test_factorial_separates_main_effects_and_interactions(self):
        runs = []
        for seed in range(3):
            for (m, s, f), method in FACTORIAL_CONDITIONS.items():
                score = 10 - m - 2*s - 3*f + m*s + seed/10
                runs.append({"method": method, "search_seed": seed, "winner": {"val_bpb": score}})
        result = factorial_effects(runs, 0)['metrics']['terminal_bpb']
        effects = {tuple(row['factors']): row['mean_effect'] for row in result['effects']}
        self.assertEqual(result['complete_seed_blocks'], 3)
        for factors, value in {('memory',): -.5, ('structure',): -1.5, ('frontier',): -3,
                               ('memory', 'structure'): 1, ('memory', 'frontier'): 0,
                               ('structure', 'frontier'): 0, ('memory', 'structure', 'frontier'): 0}.items():
            self.assertAlmostEqual(effects[factors], value)
        del runs[0]
        incomplete = factorial_effects(runs, 0)['metrics']['terminal_bpb']
        self.assertEqual(incomplete['complete_seed_blocks'], 2)
        self.assertEqual(incomplete['incomplete_seed_blocks'], 1)


if __name__ == '__main__':
    unittest.main()
