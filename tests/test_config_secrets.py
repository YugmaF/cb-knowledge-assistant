"""Start-up secrets: the app refuses to run with a missing, short or well-known secret, because
the repository is public and any default in it is a default an attacker knows."""

from __future__ import annotations

import re
import secrets
import subprocess
from pathlib import Path

import pytest

from kb_assistant.config import ConfigError, Settings, check_secret

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", [
    "",                                              # missing
    "short-secret",                                  # too short
    "x" * 31,                                        # one byte under the minimum
    "dev-only-change-me-dev-only-change-me",         # the old built-in default
    "change-me-to-a-long-random-string",             # the old .env.example value
    "REPLACE_ME_run__openssl_rand_-hex_32",          # the new placeholder
    "a-perfectly-long-value-but-change-me-please",   # long, but still a placeholder
])
def test_weak_or_default_secrets_are_rejected(value):
    with pytest.raises(ConfigError):
        check_secret("JWT_SECRET", value)


def test_a_random_secret_is_accepted():
    value = secrets.token_hex(32)
    assert check_secret("JWT_SECRET", value) == value


def test_the_error_names_the_setting_and_how_to_fix_it():
    with pytest.raises(ConfigError, match=r"JWT_SECRET.*openssl rand -hex 32"):
        check_secret("JWT_SECRET", "")


def test_settings_carry_no_built_in_jwt_secret(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    settings = Settings(_env_file=None)
    assert settings.jwt_secret == ""
    with pytest.raises(ConfigError):
        settings.validate_secrets()


def test_env_example_placeholders_are_not_usable_values():
    lines = (ROOT / ".env.example").read_text().splitlines()
    value = next(line.split("=", 1)[1] for line in lines if line.startswith("JWT_SECRET="))
    with pytest.raises(ConfigError):
        check_secret("JWT_SECRET", value)


# --- run.sh keeps working without a JWT_SECRET in the env file ---------------------------------------

def _run_ensure_secret(env_value: str | None) -> subprocess.CompletedProcess:
    """Run only run.sh's secret helpers, in a clean shell, so the test needs no services."""
    script = (ROOT / "run.sh").read_text()
    helpers = "\n".join(
        m.group(0) for m in re.finditer(r"^(random_hex|ensure_secret)\(\) \{.*?^\}", script, re.S | re.M))
    program = f'say() {{ echo "$*"; }}\n{helpers}\nensure_secret JWT_SECRET\necho "VALUE=${{JWT_SECRET}}"\n'
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin"}
    if env_value is not None:
        env["JWT_SECRET"] = env_value
    return subprocess.run(["bash", "-c", program], env=env, capture_output=True, text=True, timeout=30)


def test_run_sh_generates_a_random_secret_when_none_is_set_and_says_so():
    result = _run_ensure_secret(None)
    value = re.search(r"^VALUE=(.*)$", result.stdout, re.M)
    assert result.returncode == 0, result.stderr
    assert value and re.fullmatch(r"[0-9a-f]{64}", value.group(1))
    announcement = result.stdout.replace(value.group(0), "")
    assert "JWT_SECRET" in announcement and "generated" in announcement
    assert value.group(1) not in announcement, "the secret itself must never be printed"


def test_run_sh_keeps_a_real_secret_and_stays_quiet():
    real = secrets.token_hex(32)
    result = _run_ensure_secret(real)
    assert f"VALUE={real}" in result.stdout
    assert "generated" not in result.stdout


def test_run_sh_replaces_the_example_placeholder():
    result = _run_ensure_secret("REPLACE_ME_run__openssl_rand_-hex_32")
    assert "generated" in result.stdout
    assert "REPLACE_ME" not in re.search(r"^VALUE=(.*)$", result.stdout, re.M).group(1)
