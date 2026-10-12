import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/mlx_agent/_lab'))
from pathlib import Path

import pytest

from video_lab.database import Database


@pytest.fixture
def database(tmp_path: Path) -> Database:
    db = Database(tmp_path / "lab.sqlite3")
    db.migrate()
    return db
