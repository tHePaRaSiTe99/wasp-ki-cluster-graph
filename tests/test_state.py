import state


def test_load_returns_empty_dict_without_file():
    assert state.load() == {}


def test_save_and_load_roundtrip():
    state.save({"pages": {"https://x/personen/a/": "abc"}, "last_status": "Grüße"})
    assert state.load() == {"pages": {"https://x/personen/a/": "abc"}, "last_status": "Grüße"}
    assert not state.STATE_PATH.with_suffix(".tmp").exists()


def test_load_ignores_corrupt_file():
    state.DATA_DIR.mkdir(parents=True)
    state.STATE_PATH.write_text("{kaputt", encoding="utf-8")
    assert state.load() == {}
