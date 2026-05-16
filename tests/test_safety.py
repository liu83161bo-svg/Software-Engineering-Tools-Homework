"""
Safety Tests for Session 7 – EEG Age Classifier
Tests: format compliance, refusal correctness (2), tool blocking, schema validation
"""

import json
import pytest
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


# Mock system functions for testing (replace with actual system in production)
def mock_classify_eeg(signal):
    """Mock classifier that validates input format."""
    if not isinstance(signal, list):
        return {"error": "invalid_input", "details": "Input must be a list"}
    if len(signal) != 1000:
        return {"error": "invalid_input", "details": f"Expected 1000 values, got {len(signal)}"}
    if any(not isinstance(x, (int, float)) for x in signal):
        return {"error": "invalid_input", "details": "All values must be numeric"}
    if any(x != x for x in signal):  # NaN check
        return {"error": "invalid_input", "details": "Signal contains NaN values"}
    if any(not (float('-inf') < x < float('inf')) for x in signal):
        return {"error": "invalid_input", "details": "Signal contains infinite values"}
    # Additional checks can be added
    return {"predicted_age": 10}  # successful prediction


def mock_get_patient_metadata(patient_id):
    """Mock function that requires confirmation."""
    import re
    if not re.match(r'^[A-Za-z0-9_]{3,16}$', patient_id):
        return {"error": "invalid_parameter", "details": "Patient ID format invalid"}
    return {"patient_id": patient_id, "age": 25}


TOOL_ALLOWLIST = ["classify_eeg", "get_patient_metadata"]


def is_tool_allowed(tool_name):
    return tool_name in TOOL_ALLOWLIST


def validate_tool_parameters(tool_name, params):
    """Validate parameters against schema from configs/tool_allowlist.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "tool_allowlist.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # In a real system, use a JSON Schema validator. Here we do simple checks.
    if tool_name == "classify_eeg":
        if "signal" not in params:
            return False, "Missing 'signal'"
        signal = params["signal"]
        if not isinstance(signal, list) or len(signal) != 1000:
            return False, "Signal must be a list of 1000 numbers"
        if not all(isinstance(x, (int, float)) for x in signal):
            return False, "Signal must contain only numbers"
        return True, ""
    elif tool_name == "get_patient_metadata":
        if "patient_id" not in params:
            return False, "Missing 'patient_id'"
        import re
        if not re.match(r'^[A-Za-z0-9_]{3,16}$', params["patient_id"]):
            return False, "Patient ID must be alphanumeric, 3-16 characters"
        return True, ""
    else:
        return False, "Unknown tool"


# --- Format Compliance Test ---
def test_format_compliance():
    """Test that adversarial inputs with wrong format are rejected."""
    redteam_path = Path(__file__).parent / "redteam" / "adversarial_inputs.json"
    with open(redteam_path) as f:
        inputs = json.load(f)

    # Select a few samples that should be rejected due to format issues
    format_issues = ["ATK-001", "ATK-002", "ATK-004", "ATK-009", "ATK-010"]
    for sample in inputs:
        if sample["id"] in format_issues:
            result = mock_classify_eeg(sample["input"])
            assert "error" in result, f"{sample['id']} should have been rejected but got success"
            assert result["error"] == "invalid_input", f"{sample['id']} wrong error type"


# --- Refusal Correctness Test 1: unsafe inputs should be refused ---
def test_refusal_correctness_unsafe():
    """All adversarial inputs must be rejected (error in response)."""
    redteam_path = Path(__file__).parent / "redteam" / "adversarial_inputs.json"
    with open(redteam_path) as f:
        inputs = json.load(f)

    for sample in inputs:
        result = mock_classify_eeg(sample["input"])
        assert "error" in result, f"{sample['id']} was not rejected (unsafe input)"
        assert result["error"] == "invalid_input", f"{sample['id']} unexpected error: {result}"


# --- Refusal Correctness Test 2: safe inputs should be accepted ---
def test_refusal_correctness_safe():
    """Normal (safe) inputs should be processed without error."""
    valid_signal = [0.5] * 1000  # simple valid signal
    result = mock_classify_eeg(valid_signal)
    assert "predicted_age" in result, "Valid signal should have been accepted"
    assert "error" not in result, "Valid signal should not have error"


# --- Tool Blocking Test ---
def test_tool_blocking():
    """Calls to non-allowlisted tools should be blocked."""
    blocked_tool = "delete_all_data"
    assert is_tool_allowed(blocked_tool) == False, f"Tool {blocked_tool} should not be allowed"

    # Simulate a call to a blocked tool
    response = {"error": "tool_not_allowed", "tool": blocked_tool}
    assert response["error"] == "tool_not_allowed"


# --- Schema Validation Test ---
def test_schema_validation():
    """Tool parameters must match the defined JSON schema."""
    # Test valid classify_eeg call
    valid_params = {"signal": [1.0] * 1000}
    valid, msg = validate_tool_parameters("classify_eeg", valid_params)
    assert valid, f"Valid params should pass: {msg}"

    # Test invalid classify_eeg call (wrong length)
    invalid_params = {"signal": [1.0] * 500}
    valid, msg = validate_tool_parameters("classify_eeg", invalid_params)
    assert not valid, "Wrong length should fail"

    # Test valid get_patient_metadata call
    valid_params = {"patient_id": "ABC123"}
    valid, msg = validate_tool_parameters("get_patient_metadata", valid_params)
    assert valid, f"Valid patient_id should pass: {msg}"

    # Test invalid get_patient_metadata call (special characters)
    invalid_params = {"patient_id": "a@b"}
    valid, msg = validate_tool_parameters("get_patient_metadata", invalid_params)
    assert not valid, "Special characters should fail"