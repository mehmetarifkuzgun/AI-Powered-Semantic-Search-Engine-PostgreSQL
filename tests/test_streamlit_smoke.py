import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from conftest import ROOT  # noqa: E402


def test_streamlit_app_renders(app_modules, engine):
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.text_input][:1] == ["Enter your search query:"]
