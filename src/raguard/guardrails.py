"""Guardrails: PII screening, injection filtering, and safety checklists.

These detectors run *before* generation and on outputs. They are heuristic by
design — regex and keyword lists are auditable, fast, and dependency-free —
and the code says so openly. They are tripwires that raise the cost of
accidents, not proofs of safety; see docs/hld.md §Failure modes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    BLOCK = "block"


# Explicit ordering for severity escalation (never compare the string
# values — "block" < "info" alphabetically).
_SEVERITY_RANK = {Severity.INFO: 0, Severity.WARN: 1, Severity.BLOCK: 2}


@dataclass
class GuardrailFinding:
    """One detector hit."""

    detector: str
    severity: Severity
    message: str
    matched: str = ""


@dataclass
class GuardrailVerdict:
    """Aggregate result for a piece of text."""

    findings: list[GuardrailFinding] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(f.severity == Severity.BLOCK for f in self.findings)

    @property
    def triggered(self) -> list[GuardrailFinding]:
        return list(self.findings)


# ----------------------------------------------------------------------
# PII detector (heuristic)
# ----------------------------------------------------------------------
class PIIDetector:
    """Regex + keyword PII screen.

    Honest limitations, stated up front: pattern-based PII detection has
    recall well below 100 % (misses obfuscation, formatting variants, and
    anything not in the pattern lists) and non-trivial false-positive rates
    on non-English names and synthetic identifiers. It exists to catch the
    common cases cheaply and to *flag* text for human review — not to
    certify text as clean.
    """

    name = "pii"

    _PATTERNS: list[tuple[str, re.Pattern[str], Severity]] = [
        ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), Severity.BLOCK),
        (
            "phone",
            re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b"),
            Severity.WARN,
        ),
        ("ipv4", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), Severity.WARN),
        (
            "national_id_us_ssn",
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            Severity.BLOCK,
        ),
        (
            "credit_card",
            re.compile(r"\b(?:\d[ -]?){13,19}\b"),
            Severity.WARN,
        ),
    ]

    def __init__(self, extra_patterns: list[tuple[str, re.Pattern[str], Severity]] | None = None):
        self._patterns = list(self._PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def scan(self, text: str) -> GuardrailFinding:
        hits: list[str] = []
        worst = Severity.INFO
        for label, pattern, severity in self._patterns:
            for match in pattern.finditer(text):
                hits.append(f"{label}: {match.group(0)!r}")
                if _SEVERITY_RANK[severity] > _SEVERITY_RANK[worst]:
                    worst = severity
        message = (
            f"PII patterns matched ({len(hits)}): " + "; ".join(hits[:5])
            if hits
            else "no PII patterns matched"
        )
        return GuardrailFinding(detector=self.name, severity=worst, message=message)


# ----------------------------------------------------------------------
# Prompt-injection filter
# ----------------------------------------------------------------------
class PromptInjectionFilter:
    """Keyword/pattern screen for injection attempts in user queries.

    Catches the classic shapes — instruction override, delimiter escape,
    system-role impersonation, tool/output manipulation — with word-boundary
    patterns plus entropy-free exact phrases. Deterministic, explainable,
    and bypassable by a determined attacker; paired with least-privilege
    prompt design, it raises the bar meaningfully. Bypass attempts are
    logged, never executed.
    """

    name = "prompt_injection"

    _PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        (
            "ignore_instructions",
            re.compile(
                r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)\b", re.I
            ),
        ),
        ("forget_everything", re.compile(r"\bforget\s+everything\b", re.I)),
        (
            "system_override",
            re.compile(
                r"\b(you are|act as)\s+(now\s+)?(a|an|the)?\s*"
                r"(unrestricted|jailbroken|DAN|developer mode)\b",
                re.I,
            ),
        ),
        ("developer_tag", re.compile(r"\[/?system\]|\[/?inst\]", re.I)),
        ("delimiter_escape", re.compile(r"```\s*(system|instructions?)")),
        (
            "exfiltrate",
            re.compile(r"\b(reveal|print|repeat|output)\s+(your|the)\s+(system\s+)?prompt\b", re.I),
        ),
        ("tool_abuse", re.compile(r"\b(run|execute)\s+(this\s+)?(command|code|shell|sql)\b", re.I)),
    ]

    def scan(self, text: str) -> GuardrailFinding:
        hits = [
            f"{label}: {m.group(0)!r}"
            for label, pattern in self._PATTERNS
            for m in pattern.finditer(text)
        ]
        severity = Severity.BLOCK if hits else Severity.INFO
        message = (
            "injection pattern(s) matched: " + "; ".join(hits[:5])
            if hits
            else "no injection patterns matched"
        )
        return GuardrailFinding(detector=self.name, severity=severity, message=message)


# ----------------------------------------------------------------------
# Sensitive-domain safety checklist
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SafetyRule:
    """One item of the sensitive-domain checklist."""

    label: str
    pattern: re.Pattern[str]
    guidance: str
    severity: Severity = Severity.WARN


class SensitiveDomainGuard:
    """Checklist guard for mental-health and similar sensitive contexts.

    For sensitive-domain mode (see ADR 0001 rationale in docs/): detects
    crisis language, self-harm signals, and bias/ stereotype markers so the
    pipeline can escalate, refuse gently, or route to human review rather
    than answering naively.
    """

    name = "sensitive_domain"

    _RULES: tuple[SafetyRule, ...] = (
        SafetyRule(
            label="crisis_self_harm",
            pattern=re.compile(
                r"\b(kill (myself|me)|suicid\w+|i self[- ]?harm\b|end (my|it) (life|all)|"
                r"want to die|no reason to live|hurt(ing)? myself)\b",
                re.I,
            ),
            guidance=(
                "Crisis/self-harm signal detected. A production system must "
                "surface vetted crisis resources and route to human review, "
                "never improvise therapy. See docs/hld.md §Failure modes."
            ),
            severity=Severity.BLOCK,
        ),
        SafetyRule(
            label="biased_assumption",
            pattern=re.compile(
                r"\b(all|every|no)\s+(men|women|patients?|therapists?|people\s+with)\b|"
                r"\b(they'?re all|those people)\b",
                re.I,
            ),
            guidance=(
                "Overgeneralising/bias-prone phrasing detected; responses "
                "must avoid stereotyping groups of people."
            ),
            severity=Severity.WARN,
        ),
        SafetyRule(
            label="medical_advice_risk",
            pattern=re.compile(
                r"\b(should i stop|can i quit)\s+(my\s+)?(medication|meds|antidepress\w+|"
                r"therapy|treatment)\b|\bwhat dose\b|\bdiagnos(e|is) me\b",
                re.I,
            ),
            guidance=(
                "Medication/diagnosis-seeking query detected; answer must "
                "defer to qualified professionals and never give dosing or "
                "diagnostic advice."
            ),
            severity=Severity.WARN,
        ),
    )

    def scan(self, text: str) -> list[GuardrailFinding]:
        findings: list[GuardrailFinding] = []
        for rule in self._RULES:
            match = rule.pattern.search(text)
            if match:
                findings.append(
                    GuardrailFinding(
                        detector=f"{self.name}:{rule.label}",
                        severity=rule.severity,
                        message=f"{rule.guidance} (matched: {match.group(0)!r})",
                        matched=match.group(0),
                    )
                )
        return findings


# ----------------------------------------------------------------------
# Facade
# ----------------------------------------------------------------------
class GuardrailSuite:
    """Runs all configured detectors over a question (and optionally output)."""

    def __init__(
        self,
        pii: PIIDetector | None = None,
        injection: PromptInjectionFilter | None = None,
        sensitive: SensitiveDomainGuard | None = None,
        sensitive_mode: bool = False,
    ) -> None:
        self.pii = pii or PIIDetector()
        self.injection = injection or PromptInjectionFilter()
        self.sensitive = sensitive or SensitiveDomainGuard()
        self.sensitive_mode = sensitive_mode

    def check_question(self, question: str) -> GuardrailVerdict:
        """Screen an incoming question before retrieval/generation."""
        findings = [
            self.pii.scan(question),
            self.injection.scan(question),
        ]
        if self.sensitive_mode:
            findings.extend(self.sensitive.scan(question))
        return GuardrailVerdict(findings=findings)

    def check_output(self, answer: str) -> GuardrailVerdict:
        """Screen a generated answer before it is returned."""
        findings: list[GuardrailFinding] = [self.pii.scan(answer)]
        if self.sensitive_mode:
            findings.extend(self.sensitive.scan(answer))
        return GuardrailVerdict(findings=findings)
