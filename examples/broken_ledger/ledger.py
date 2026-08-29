"""A tiny double-entry ledger.

The implementation in this file is intentionally wrong. Run
`python -m unittest -v` in this directory to see the failing tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Line:
    account: str
    cents: int
    voided: bool = False


class Ledger:
    def __init__(self) -> None:
        self._lines: list[Line] = []

    def record(self, account: str, dollars: float) -> None:
        # Stores dollars instead of integer cents, so 10.10 becomes 10.1
        # rather than 1010, and binary floats make 0.10 + 0.20 != 0.30.
        self._lines.append(Line(account, dollars))  # type: ignore[arg-type]

    def void(self, index: int) -> None:
        self._lines[index].voided = True

    def balance(self, account: str) -> int:
        # Counts voided lines and returns a float when record() stored dollars.
        total = 0
        for line in self._lines:
            if line.account == account:
                total += line.cents
        return total

    def is_balanced(self) -> bool:
        return sum(line.cents for line in self._lines) == 0
