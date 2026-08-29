# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from loguru import logger
from pydantic_models import Test
from typing import Optional, List, ClassVar, Tuple
from .parser import Parser
from pydantic import ValidationError
from shared import failure_happened
import json
import re


CATEGORY = "parameter_support"
OWNER = "tt-shield"

# tt-inference-server emits spec-test results in two shapes:
#   * the original flat one, under a top-level ``parameter_support_tests`` key
#   * the v2 one, as ``sections[]`` blocks tagged ``kind: "spec_tests"``
# Both are read here; the v2 blocks carry no per-test ids or timestamps, so a
# few fields are derived rather than copied (see _block_window).
SPEC_TESTS_KIND = "spec_tests"
UNKNOWN_TS = "9999-12-31T23:59:59Z"
UNKNOWN_FILEPATH = "unknown"

PASS_STATUSES = ("passed", "success", "pass", "ok")
FAIL_STATUSES = ("failed", "failure", "fail", "error")
SKIP_STATUSES = ("skipped", "skip")

# v2 statuses are rendered for humans and carry a leading glyph ("✅ PASSED"),
# so a bare .lower() never matches the plain words above.
_STATUS_TRIM_RE = re.compile(r"^[^0-9A-Za-z]+|[^0-9A-Za-z]+$")


def _normalize_status(value) -> str:
    """Lowercase a status string, dropping any surrounding glyphs."""
    if not isinstance(value, str):
        return ""
    return _STATUS_TRIM_RE.sub("", value.strip()).lower()


def _resolve_outcome(status, success_flag) -> Tuple[bool, bool]:
    """Return ``(success, skipped)`` from a status string.

    Falls back to the block's own boolean ``success`` when the status string is
    missing or unrecognised, which is how suite-level blocks report themselves.
    """
    normalized = _normalize_status(status)
    if normalized in SKIP_STATUSES:
        return False, True
    if normalized in PASS_STATUSES:
        return True, False
    if normalized in FAIL_STATUSES:
        return False, False
    if isinstance(success_flag, bool):
        return success_flag, False
    return False, False


def _block_window(generated_at, elapsed_seconds) -> Tuple[str, str]:
    """Return ``(start_ts, end_ts)`` for a v2 spec-test block.

    v2 blocks carry no per-test timestamps — only the report's ``generated_at``
    and the block's own ``elapsed_seconds`` — so every test in a block shares
    the block's window. An approximation, but a far more useful one than the
    ``9999-12-31`` sentinel the flat shape falls back to.
    """
    if not generated_at:
        return UNKNOWN_TS, UNKNOWN_TS
    try:
        end = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
    except ValueError:
        logger.warning(f"Could not parse generated_at {generated_at!r}; falling back to sentinel timestamps")
        return UNKNOWN_TS, UNKNOWN_TS
    try:
        elapsed = max(float(elapsed_seconds), 0.0)
    except (TypeError, ValueError):
        elapsed = 0.0
    return (end - timedelta(seconds=elapsed)).isoformat(), end.isoformat()


@dataclass(frozen=True)
class ParameterSupportTestConfig:
    _PARAM_SUPPORT_TEST_PROPS: ClassVar[tuple] = ("model_name", "model_impl", "device", "endpoint_url")
    model_name: str = "unknown_model"
    model_impl: str = "unknown_impl"
    device: str = "unknown_device"
    endpoint_url: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "ParameterSupportTestConfig":
        return cls(**{k: data[k] for k in cls._PARAM_SUPPORT_TEST_PROPS if k in data})


@dataclass(frozen=True)
class ParameterSupportTestTags:
    type: str = "parameter_support_test"


class ParameterSupportTestParser(Parser):
    """Parser for spec / parameter support test JSON report files."""

    def can_parse(self, filepath: str):
        """Check if file is JSON and carries spec test results in either shape."""

        if not filepath.endswith(".json"):
            return False

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.error(f"Failed to load JSON from {filepath}: {e}")
            return False

        return self._has_flat_results(data) or bool(self._spec_test_blocks(data))

    @staticmethod
    def _has_flat_results(data) -> bool:
        if not isinstance(data, dict):
            return False
        param_support_tests = data.get("parameter_support_tests")
        return isinstance(param_support_tests, dict) and "results" in param_support_tests

    @staticmethod
    def _spec_test_blocks(data) -> List[dict]:
        if not isinstance(data, dict):
            return []
        return [
            block
            for block in (data.get("sections") or [])
            if isinstance(block, dict) and block.get("kind") == SPEC_TESTS_KIND and isinstance(block.get("data"), dict)
        ]

    def parse(
        self,
        filepath: str,
        project: Optional[str] = None,
        github_job_id: Optional[int] = None,
    ) -> List[Test]:
        logger.info(f"Parsing spec tests from {filepath}")

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load JSON from {filepath}: {e}")
            return []

        # The two shapes are mutually exclusive in practice; preferring the flat
        # one keeps older reports parsing exactly as they always have.
        if self._has_flat_results(data):
            tests = self._parse_flat(data, filepath)
        else:
            tests = self._parse_sections(data)

        logger.info(f"Parsed {len(tests)} spec tests from {filepath}")
        return tests

    def _parse_flat(self, data: dict, filepath: str) -> List[Test]:
        metadata = data.get("metadata", {})
        param_support_tests = data.get("parameter_support_tests", {})
        if not param_support_tests or "results" not in param_support_tests:
            logger.warning(f"No parameter support test results found in {filepath}")
            return []
        # Merge metadata into param_support_tests without allowing it to override the actual test results
        filtered_metadata = {k: v for k, v in metadata.items() if k != "results"}
        param_support_tests = {
            **param_support_tests,
            **filtered_metadata,
        }  # metadata values take precedence (except 'results')

        tests = []
        for test_group_name, test_case_results in param_support_tests.get("results", {}).items():
            for test_case in test_case_results:
                test = self._create_test_from_case(
                    param_support_tests=param_support_tests,
                    test_group_name=test_group_name,
                    test_case=test_case,
                )
                if test:
                    tests.append(test)
        return tests

    def _parse_sections(self, data: dict) -> List[Test]:
        """Build tests from v2 ``sections[kind="spec_tests"]`` blocks."""
        metadata = data.get("metadata") or {}
        generated_at = metadata.get("generated_at")
        # Metadata wins over the block payload, matching the flat path.
        shared_config = {k: v for k, v in metadata.items() if k != "results"}

        tests = []
        for block in self._spec_test_blocks(data):
            block_data = block["data"]
            group = block_data.get("test_name") or block.get("title") or "unknown"
            start_ts, end_ts = _block_window(generated_at, block_data.get("elapsed_seconds"))
            config_source = {**block_data, **shared_config}

            detailed_results = block_data.get("detailed_test_results")
            if isinstance(detailed_results, list) and detailed_results:
                for test_case in detailed_results:
                    if not isinstance(test_case, dict):
                        continue
                    test = self._create_test_from_block(
                        config_source=config_source,
                        group=group,
                        test_case_name=(test_case.get("parametrization") or test_case.get("test_case") or "unknown"),
                        status=test_case.get("status"),
                        success_flag=test_case.get("success"),
                        message=test_case.get("message", ""),
                        start_ts=start_ts,
                        end_ts=end_ts,
                    )
                    if test:
                        tests.append(test)
                continue

            # Blocks with no per-test breakdown (e.g. LoggerForkSafetyTest)
            # contribute a single row for the suite as a whole.
            test = self._create_test_from_block(
                config_source=config_source,
                group=group,
                test_case_name=group,
                status=block_data.get("status"),
                success_flag=block_data.get("success"),
                message=str(block_data.get("child_result") or block_data.get("description") or ""),
                start_ts=start_ts,
                end_ts=end_ts,
            )
            if test:
                tests.append(test)
        return tests

    def _create_test_from_block(
        self,
        config_source: dict,
        group: str,
        test_case_name: str,
        status,
        success_flag,
        message: str,
        start_ts: str,
        end_ts: str,
    ) -> Optional[Test]:
        """Convert one v2 spec-test result to a Test object."""

        success, skipped = _resolve_outcome(status, success_flag)

        error_message = None
        if not success or skipped:
            error_message = message or None

        full_test_name = f"{group}::{test_case_name}" if test_case_name != group else group
        config = ParameterSupportTestConfig.from_dict(config_source)
        tags = ParameterSupportTestTags()

        try:
            return Test(
                test_start_ts=start_ts,
                test_end_ts=end_ts,
                test_case_name=test_case_name,
                filepath=UNKNOWN_FILEPATH,
                category=CATEGORY,
                group=group,
                owner=OWNER,
                error_message=error_message,
                success=success,
                skipped=skipped,
                full_test_name=full_test_name,
                config=asdict(config),
                tags=asdict(tags),
            )
        except ValidationError as e:
            failure_happened()
            logger.error(f"Validation error creating Test for {test_case_name}: {e}")
            return None

    def _create_test_from_case(
        self,
        param_support_tests: dict,
        test_group_name: str,
        test_case: dict,
    ) -> Optional[Test]:
        """Convert a parameter support test case to a Test object."""

        test_start_ts = test_case.get("test_start_ts", UNKNOWN_TS)
        test_end_ts = test_case.get("test_end_ts", UNKNOWN_TS)
        test_case_name = test_case.get("test_node_name", "unknown")

        if "test_id" in test_case:
            filepath = test_case["test_id"].split("::")[0]
        else:
            filepath = UNKNOWN_FILEPATH

        status = test_case.get("status", "unknown").lower()
        message = test_case.get("message", "")
        success = status in PASS_STATUSES
        failed = status in FAIL_STATUSES
        skipped = status in SKIP_STATUSES

        error_message = None
        if failed or skipped:
            error_message = message

        full_test_name = test_case.get("test_id", "unknown")
        config = ParameterSupportTestConfig.from_dict(param_support_tests)
        tags = ParameterSupportTestTags()

        try:
            return Test(
                test_start_ts=test_start_ts,
                test_end_ts=test_end_ts,
                test_case_name=test_case_name,
                filepath=filepath,
                category=CATEGORY,
                group=test_group_name,
                owner=OWNER,
                error_message=error_message,
                success=success,
                skipped=skipped,
                full_test_name=full_test_name,
                config=asdict(config),
                tags=asdict(tags),
            )
        except ValidationError as e:
            failure_happened()
            logger.error(f"Validation error creating Test for {test_case_name}: {e}")
            return None
