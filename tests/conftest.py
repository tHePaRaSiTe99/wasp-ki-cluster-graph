"""Gemeinsame Fixtures. Kein Test darf die echten Dateien unter data/ oder web/ anfassen."""
import pytest

import state


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Leitet data/state.json für jeden Test in ein temporäres Verzeichnis um."""
    data_dir = tmp_path / "state"
    monkeypatch.setattr(state, "DATA_DIR", data_dir)
    monkeypatch.setattr(state, "STATE_PATH", data_dir / "state.json")
    return data_dir / "state.json"
