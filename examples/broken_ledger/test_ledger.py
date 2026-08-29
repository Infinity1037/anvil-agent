import unittest

from ledger import Ledger


class LedgerTests(unittest.TestCase):
    def test_amounts_are_integer_cents(self) -> None:
        ledger = Ledger()
        ledger.record("cash", 10.10)
        ledger.record("revenue", -10.10)
        self.assertEqual(ledger.balance("cash"), 1010)
        self.assertIsInstance(ledger.balance("cash"), int)
        self.assertTrue(ledger.is_balanced())

    def test_voided_lines_are_excluded(self) -> None:
        ledger = Ledger()
        ledger.record("cash", 5.00)
        ledger.record("revenue", -5.00)
        ledger.void(0)
        ledger.void(1)
        self.assertEqual(ledger.balance("cash"), 0)
        self.assertTrue(ledger.is_balanced())

    def test_unbalanced_entry_is_detected(self) -> None:
        ledger = Ledger()
        ledger.record("cash", 1.00)
        self.assertFalse(ledger.is_balanced())

    def test_binary_float_rounding_does_not_break_cents(self) -> None:
        ledger = Ledger()
        ledger.record("cash", 0.10)
        ledger.record("cash", 0.20)
        ledger.record("expense", -0.30)
        self.assertEqual(ledger.balance("cash"), 30)
        self.assertTrue(ledger.is_balanced())


if __name__ == "__main__":
    unittest.main()
