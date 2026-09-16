from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_starts_without_exception():
    app = AppTest.from_file(str(ROOT / "app" / "역전점_앱.py"))
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "⚾ KBO 투수 피로 신호 탐색"
    assert len(app.selectbox) >= 5
