"""Removing any billed component or rounding time upward must fail these tests."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    from rental_budget import plan_budget
except ImportError:
    plan_budget = None


class RentalBudgetTests(unittest.TestCase):
    def quote(self, **updates):
        result = dict(cap_usd='100', spent_usd='10', reserve_usd='5',
                      compute_usd_hour='30', storage_usd_hour='2',
                      ingress_usd_unit='1', egress_usd_unit='4',
                      transfer_unit_bytes=1000, ingress_bytes=1000,
                      egress_bytes=2000, billing_quantum_seconds=60,
                      teardown_seconds=120)
        result.update(updates)
        return result

    def test_billed_time_includes_storage_transfer_spent_and_teardown(self):
        self.assertIsNotNone(plan_budget, 'rental cost planner is missing')
        plan = plan_budget(self.quote())
        # 100-10-5-1-8 = 76; 76/32 hours = 8550 seconds.
        # Round DOWN to 8520 paid seconds, of which final 120 are teardown.
        self.assertEqual(plan['max_billed_seconds'], 8520)
        self.assertEqual(plan['work_deadline_seconds'], 8400)
        self.assertEqual(plan['transfer_budget_usd'], '9')
        self.assertFalse(plan['enforces_billing_stop'])

    def test_no_time_or_unknown_numbers_rejected(self):
        self.assertIsNotNone(plan_budget)
        for changes in [dict(spent_usd='100'), dict(reserve_usd='99'),
                        dict(teardown_seconds=10000), dict(compute_usd_hour='NaN'),
                        dict(storage_usd_hour='-1'), dict(egress_bytes=True),
                        dict(transfer_unit_bytes=0), dict(billing_quantum_seconds=0),
                        dict(compute_usd_hour=30.1)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                plan_budget(self.quote(**changes))

    def test_decimal_quote_never_rounds_deadline_up(self):
        self.assertIsNotNone(plan_budget)
        plan = plan_budget(self.quote(cap_usd='1', spent_usd='0', reserve_usd='0',
                           compute_usd_hour='0.7', storage_usd_hour='0',
                           ingress_bytes=0, egress_bytes=0,
                           billing_quantum_seconds=1, teardown_seconds=1))
        self.assertEqual(plan['max_billed_seconds'], 5142)
        self.assertEqual(plan['work_deadline_seconds'], 5141)


if __name__ == '__main__':
    unittest.main()
