# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import json
import tempfile
from pathlib import Path
from parsers.parameter_support_test_parser import ParameterSupportTestParser


@pytest.fixture
def sample_parameter_support_json():
    """Create a sample parameter support test JSON file."""
    data = {
        "metadata": {
            "model_name": "Llama-3.1-8B-Instruct",
            "model_impl": "tt-transformers",
            "device": "n300",
        },
        "parameter_support_tests": {
            "endpoint_url": "http://127.0.0.1:8000/v1/chat/completions",
            "model_name": "Llama-3.1-8B-Instruct",
            "model_impl": "tt-transformers",
            "results": {
                "test_n": [
                    {"status": "failed", "message": "Connection refused", "test_node_name": "test_n[2]"},
                    {"status": "passed", "message": "", "test_node_name": "test_n[3]"},
                ],
                "test_max_tokens": [
                    {"status": "passed", "message": "max_tokens=2048 supported", "test_node_name": "test_max_tokens[5]"}
                ],
            },
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink()


def test_can_parse_parameter_support_json(sample_parameter_support_json):
    parser = ParameterSupportTestParser()
    assert parser.can_parse(sample_parameter_support_json) is True


def test_cannot_parse_non_json():
    parser = ParameterSupportTestParser()
    assert parser.can_parse("test.xml") is False


def test_parse_parameter_support_tests(sample_parameter_support_json):
    parser = ParameterSupportTestParser()
    tests = parser.parse(sample_parameter_support_json)

    assert len(tests) == 3

    assert tests[0].config["endpoint_url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert tests[0].config["model_name"] == "Llama-3.1-8B-Instruct"
    assert tests[0].config["model_impl"] == "tt-transformers"
    assert tests[0].config["device"] == "n300"
    assert tests[0].test_case_name == "test_n[2]"
    assert tests[0].success is False
    assert tests[0].error_message == "Connection refused"
    assert tests[0].category == "parameter_support"
    assert tests[0].owner == "tt-shield"
    assert tests[0].group == "test_n"
    assert tests[0].tags["type"] == "parameter_support_test"

    assert tests[1].config["endpoint_url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert tests[1].config["model_name"] == "Llama-3.1-8B-Instruct"
    assert tests[1].config["model_impl"] == "tt-transformers"
    assert tests[1].config["device"] == "n300"
    assert tests[1].test_case_name == "test_n[3]"
    assert tests[1].success is True
    assert tests[1].error_message is None
    assert tests[1].category == "parameter_support"
    assert tests[1].owner == "tt-shield"
    assert tests[1].group == "test_n"
    assert tests[1].tags["type"] == "parameter_support_test"

    assert tests[2].config["endpoint_url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert tests[2].config["model_name"] == "Llama-3.1-8B-Instruct"
    assert tests[2].config["model_impl"] == "tt-transformers"
    assert tests[2].config["device"] == "n300"
    assert tests[2].test_case_name == "test_max_tokens[5]"
    assert tests[2].success is True
    assert tests[2].error_message is None
    assert tests[2].category == "parameter_support"
    assert tests[2].owner == "tt-shield"
    assert tests[2].group == "test_max_tokens"
    assert tests[2].tags["type"] == "parameter_support_test"


@pytest.fixture
def sample_spec_tests_sections_json():
    """A v2 report: spec test results as sections[] blocks, as tt-inference-server emits them."""
    data = {
        "metadata": {
            "model_name": "google/diffusiongemma-26B-A4B-it",
            "device": "P300X2",
            "model_impl": "diffusion-gemma",
            "generated_at": "2026-08-27T20:20:49+00:00",
        },
        "sections": [
            {
                "kind": "benchmarks",
                "data": {"mean_ttft_ms": 12350.428},
                "title": "vLLM Benchmark",
            },
            {
                "kind": "spec_tests",
                "task_type": "unit",
                "title": "Logger Fork Safety",
                "data": {
                    "success": True,
                    "child_result": "OK",
                    "status": "pass",
                    "elapsed_seconds": 0.028281036764383316,
                    "test_name": "LoggerForkSafetyTest",
                    "description": "Test for logging fork safety to prevent deadlocks",
                },
            },
            {
                "kind": "spec_tests",
                "task_type": "functional",
                "title": "Vllm Diffusiongemma",
                "data": {
                    "endpoint_url": "http://127.0.0.1:8000/v1/chat/completions",
                    "model_name": "google/diffusiongemma-26B-A4B-it",
                    "detailed_test_results": [
                        {
                            "test_case": "test_exact_context_and_canvas_fit",
                            "parametrization": "test_exact_context_and_canvas_fit",
                            "status": "✅ PASSED",
                            "message": "",
                        },
                        {
                            "test_case": "test_unsupported_response_parameter",
                            "parametrization": "test_unsupported_response_parameter[n-2]",
                            "status": "❌ FAILED",
                            "message": "n=2 not supported",
                        },
                        {
                            "test_case": "test_optional_capability",
                            "parametrization": "test_optional_capability",
                            "status": "➖ SKIPPED",
                            "message": "not applicable on this device",
                        },
                    ],
                    "success": True,
                    "status": "pass",
                    "elapsed_seconds": 199.57880044076592,
                    "test_name": "VLLMDiffusionGemmaParamConformanceTest",
                },
            },
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink()


def test_can_parse_spec_tests_sections(sample_spec_tests_sections_json):
    parser = ParameterSupportTestParser()
    assert parser.can_parse(sample_spec_tests_sections_json) is True


def test_cannot_parse_report_without_spec_tests_sections():
    """A report carrying only benchmark/eval sections is not ours to parse."""
    data = {
        "metadata": {"model_name": "Qwen3-32B"},
        "sections": [{"kind": "benchmarks", "data": {"mean_ttft_ms": 1.0}}],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        parser = ParameterSupportTestParser()
        assert parser.can_parse(temp_path) is False
    finally:
        Path(temp_path).unlink()


def test_parse_spec_tests_sections(sample_spec_tests_sections_json):
    parser = ParameterSupportTestParser()
    tests = parser.parse(sample_spec_tests_sections_json)

    # 1 suite-level block (no per-test breakdown) + 3 detailed results
    assert len(tests) == 4

    suite = tests[0]
    assert suite.test_case_name == "LoggerForkSafetyTest"
    assert suite.group == "LoggerForkSafetyTest"
    assert suite.full_test_name == "LoggerForkSafetyTest"
    assert suite.success is True
    assert suite.skipped is False
    assert suite.error_message is None
    assert suite.category == "parameter_support"
    assert suite.owner == "tt-shield"
    assert suite.tags["type"] == "parameter_support_test"


def test_glyph_prefixed_statuses_are_understood(sample_spec_tests_sections_json):
    """v2 renders statuses as '✅ PASSED' — a bare .lower() would mark every test failed."""
    parser = ParameterSupportTestParser()
    tests = parser.parse(sample_spec_tests_sections_json)
    by_name = {t.test_case_name: t for t in tests}

    passed = by_name["test_exact_context_and_canvas_fit"]
    assert passed.success is True
    assert passed.skipped is False
    assert passed.error_message is None

    failed = by_name["test_unsupported_response_parameter[n-2]"]
    assert failed.success is False
    assert failed.skipped is False
    assert failed.error_message == "n=2 not supported"

    skipped = by_name["test_optional_capability"]
    assert skipped.success is False
    assert skipped.skipped is True
    assert skipped.error_message == "not applicable on this device"


def test_spec_tests_sections_naming_and_config(sample_spec_tests_sections_json):
    parser = ParameterSupportTestParser()
    tests = parser.parse(sample_spec_tests_sections_json)
    by_name = {t.test_case_name: t for t in tests}

    case = by_name["test_unsupported_response_parameter[n-2]"]
    assert case.group == "VLLMDiffusionGemmaParamConformanceTest"
    assert case.full_test_name == ("VLLMDiffusionGemmaParamConformanceTest::test_unsupported_response_parameter[n-2]")
    # Metadata supplies model/device/impl; the block supplies the endpoint.
    assert case.config["model_name"] == "google/diffusiongemma-26B-A4B-it"
    assert case.config["device"] == "P300X2"
    assert case.config["model_impl"] == "diffusion-gemma"
    assert case.config["endpoint_url"] == "http://127.0.0.1:8000/v1/chat/completions"


def test_spec_tests_sections_derive_timestamps_from_report(sample_spec_tests_sections_json):
    """Timestamps come from generated_at minus the block's elapsed time, not the 9999 sentinel."""
    parser = ParameterSupportTestParser()
    tests = parser.parse(sample_spec_tests_sections_json)
    by_name = {t.test_case_name: t for t in tests}

    suite = by_name["LoggerForkSafetyTest"]
    assert suite.test_end_ts.year == 2026
    assert suite.test_start_ts.year == 2026
    assert suite.test_start_ts <= suite.test_end_ts

    # The 199.6s block starts meaningfully before it ends.
    case = by_name["test_exact_context_and_canvas_fit"]
    assert (case.test_end_ts - case.test_start_ts).total_seconds() == pytest.approx(199.58, abs=0.01)


def test_spec_tests_sections_without_generated_at_fall_back_to_sentinel():
    data = {
        "metadata": {"model_name": "Qwen3-32B"},
        "sections": [
            {
                "kind": "spec_tests",
                "data": {"status": "pass", "success": True, "test_name": "LoggerForkSafetyTest"},
            }
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        parser = ParameterSupportTestParser()
        tests = parser.parse(temp_path)
        assert len(tests) == 1
        assert tests[0].test_start_ts.year == 9999
        assert tests[0].success is True
    finally:
        Path(temp_path).unlink()


def test_suite_block_falls_back_to_success_flag_when_status_missing():
    """Blocks that report only a boolean still resolve correctly."""
    data = {
        "metadata": {"model_name": "Qwen3-32B", "generated_at": "2026-08-27T20:20:49+00:00"},
        "sections": [
            {
                "kind": "spec_tests",
                "data": {"success": False, "test_name": "BrokenSuite", "child_result": "child exited 1"},
            }
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        parser = ParameterSupportTestParser()
        tests = parser.parse(temp_path)
        assert len(tests) == 1
        assert tests[0].success is False
        assert tests[0].error_message == "child exited 1"
    finally:
        Path(temp_path).unlink()


def test_parse_empty_results():
    data = {"parameter_support_tests": {"endpoint_url": "http://test", "model_name": "test_model", "results": {}}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        parser = ParameterSupportTestParser()
        tests = parser.parse(temp_path)
        assert len(tests) == 0
    finally:
        Path(temp_path).unlink()


def test_metadata_takes_precedence():
    """Test that metadata values override parameter_support_tests values."""
    data = {
        "metadata": {
            "model_name": "Metadata-Model",
            "device": "n150",
        },
        "parameter_support_tests": {
            "endpoint_url": "http://127.0.0.1:8000/v1/chat/completions",
            "model_name": "Original-Model",
            "model_impl": "tt-transformers",
            "device": "n300",
            "results": {
                "test_n": [
                    {"status": "passed", "message": "", "test_node_name": "test_n[1]"},
                ],
            },
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        parser = ParameterSupportTestParser()
        tests = parser.parse(temp_path)

        assert len(tests) == 1
        # Metadata values should override parameter_support_tests values
        assert tests[0].config["model_name"] == "Metadata-Model"
        assert tests[0].config["device"] == "n150"
        # Values not in metadata should come from parameter_support_tests
        assert tests[0].config["endpoint_url"] == "http://127.0.0.1:8000/v1/chat/completions"
        assert tests[0].config["model_impl"] == "tt-transformers"
    finally:
        Path(temp_path).unlink()


def test_metadata_results_does_not_override_actual_results():
    """Test that a 'results' key in metadata does not interfere with actual test results parsing."""
    data = {
        "metadata": {
            "model_name": "Metadata-Model",
            "results": {
                "fake_test": [
                    {"status": "passed", "message": "", "test_node_name": "fake_test[1]"},
                ],
            },
        },
        "parameter_support_tests": {
            "endpoint_url": "http://127.0.0.1:8000/v1/chat/completions",
            "model_name": "Original-Model",
            "model_impl": "tt-transformers",
            "results": {
                "real_test": [
                    {"status": "passed", "message": "", "test_node_name": "real_test[1]"},
                    {"status": "failed", "message": "Error", "test_node_name": "real_test[2]"},
                ],
            },
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        parser = ParameterSupportTestParser()
        tests = parser.parse(temp_path)

        # Should parse results from parameter_support_tests, not metadata
        assert len(tests) == 2
        assert tests[0].test_case_name == "real_test[1]"
        assert tests[0].group == "real_test"
    finally:
        Path(temp_path).unlink()
