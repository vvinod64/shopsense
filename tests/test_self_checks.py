"""Every module's offline self-check, run by pytest in its own process (some set environment variables).

The same checks run from the command line with `python -m shopsense.<module> --offline`.
"""
import subprocess
import sys

import pytest

MODULES = ["shopsense.mock_api", "shopsense.reliability", "shopsense.guardrails",
           "shopsense.m1.parser", "shopsense.m2.agent", "shopsense.m3.memory", "shopsense.m3.rag",
           "shopsense.m4.hybrid", "shopsense.m4.eval", "shopsense.m5.workflow", "shopsense.m6.team",
           "shopsense.m7.observability", "shopsense.m7.canary", "shopsense.m8.eval", "shopsense.m8.api"]


@pytest.mark.parametrize("module", MODULES)
def test_offline_self_check(module):
    run = subprocess.run([sys.executable, "-c", f"from {module} import self_check; self_check()"],
                         capture_output=True, text=True, timeout=600,
                         env={**__import__("os").environ, "SHOPSENSE_TRACING": "0", "SHOPSENSE_LLM_COMMITMENT_CHECK": "off",
                              "SHOPSENSE_INJECTION_CLASSIFIER": "off",
                              "SHOPSENSE_INJECTION_JUDGE": "off"})
    assert run.returncode == 0, f"{module} self-check failed:\n{run.stdout[-2000:]}\n{run.stderr[-3000:]}"
