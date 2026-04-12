from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TRANSLATION_ROOT = ROOT / "deep_research" / "asr-v2" / "translation"
if str(TRANSLATION_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSLATION_ROOT))


MODULE_PATH = TRANSLATION_ROOT / "stage_04_deep_agents" / "01_translation_deep_agent.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_translation_deep_agents", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_translation_deep_agent_requires_install_or_returns_agent():
    module = _load_module()

    try:
        agent = module.build_translation_deep_agent()
    except RuntimeError as exc:
        assert "uv add deepagents" in str(exc)
    else:
        assert hasattr(agent, "invoke")
