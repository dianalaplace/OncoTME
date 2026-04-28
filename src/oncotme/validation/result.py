"""CheckResult / ValidationReport — примитивы накопления результатов проверок.

Политика серьёзности:
- ``PASS`` — проверка прошла, ничего не делаем.
- ``WARN`` — подозрительно, но не блокер (пример: сильно несбалансированные классы,
  extreme-skew экспрессии).
- ``FAIL`` — нарушен инвариант пайплайна, дальше ехать нельзя
  (пример: NaN в таргете, утечка test→train).

``ValidationReport.assert_no_failures()`` поднимает исключение, если есть хотя бы
один FAIL. Это используется в CLI-скриптах как hard gate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    """Результат одной проверки.

    Attributes
    ----------
    name
        Короткое имя проверки (``cohort.no_nan_in_endpoints``).
    status
        PASS / WARN / FAIL / SKIP.
    message
        Человекочитаемое объяснение (одно-два предложения).
    evidence
        Произвольные сериализуемые данные (shape, counts, p-values) — для отчёта
        и последующей отладки.
    """

    name: str
    status: CheckStatus
    message: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class Check:
    """Ленивая проверка: имя + вызываемое, возвращающее CheckResult.

    Нужно, чтобы suite мог показать план проверок до их запуска и чтобы
    отдельные проверки можно было skipped по фильтру (например, ``--only cohort.*``).
    """

    name: str
    fn: Callable[[], CheckResult]
    tags: List[str] = field(default_factory=list)

    def run(self) -> CheckResult:
        try:
            result = self.fn()
        except Exception as exc:  # noqa: BLE001 — проверка ничего не должна ронять наружу
            return CheckResult(
                name=self.name,
                status=CheckStatus.FAIL,
                message=f"Check raised {type(exc).__name__}: {exc}",
                evidence={"exception_type": type(exc).__name__},
            )
        # гарантируем имя
        if result.name != self.name:
            result.name = self.name
        return result


@dataclass
class ValidationReport:
    """Аккумулятор результатов проверок + сериализация + hard-gate."""

    results: List[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    def extend(self, results: List[CheckResult]) -> None:
        self.results.extend(results)

    # ---- аналитика -----------------------------------------------------

    def counts(self) -> Dict[str, int]:
        c = {s.value: 0 for s in CheckStatus}
        for r in self.results:
            c[r.status.value] += 1
        return c

    def failures(self) -> List[CheckResult]:
        return [r for r in self.results if r.status == CheckStatus.FAIL]

    def warnings(self) -> List[CheckResult]:
        return [r for r in self.results if r.status == CheckStatus.WARN]

    def all_passed(self) -> bool:
        return not self.failures()

    def assert_no_failures(self) -> None:
        fails = self.failures()
        if fails:
            msg = "Validation failed:\n" + "\n".join(
                f"  [FAIL] {r.name}: {r.message}" for r in fails
            )
            raise AssertionError(msg)

    # ---- сериализация --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counts": self.counts(),
            "results": [r.as_row() for r in self.results],
        }

    def to_json(self, path: Optional[str | Path] = None) -> str:
        data = json.dumps(self.to_dict(), indent=2, default=str)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(data)
        return data

    def to_text(self) -> str:
        lines: List[str] = []
        counts = self.counts()
        lines.append(
            f"Validation report: PASS={counts['PASS']} WARN={counts['WARN']} "
            f"FAIL={counts['FAIL']} SKIP={counts['SKIP']} (total={len(self.results)})"
        )
        # ширина колонки имени — подгоняем
        width = max((len(r.name) for r in self.results), default=4)
        for r in self.results:
            badge = {
                CheckStatus.PASS: "  OK  ",
                CheckStatus.WARN: " WARN ",
                CheckStatus.FAIL: " FAIL ",
                CheckStatus.SKIP: " SKIP ",
            }[r.status]
            lines.append(f"[{badge}] {r.name.ljust(width)} — {r.message}")
        return "\n".join(lines)


# ---- вспомогалка ---------------------------------------------------------


def asdict_result(r: CheckResult) -> Dict[str, Any]:
    return asdict(r)
