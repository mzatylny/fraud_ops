from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_exceptions():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Live traffic",
        "Analyst review",
        "Monitoring",
        "Model evaluation",
        "Dataset explorer",
    ]
