import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["PIA_AUDIT_DIR"] = tempfile.mkdtemp(prefix="pia_audit_")


@pytest.fixture(scope="session", autouse=True)
def warehouse():
    """Use an existing Field Performance Hub warehouse, or build one into a temp dir."""
    from pia.settings import warehouse_path
    if "PIA_WAREHOUSE" not in os.environ and warehouse_path().exists():
        return warehouse_path()
    tmp = Path(tempfile.mkdtemp(prefix="pia_wh_"))
    os.environ.update({"FPH_DATA_DIR": str(tmp / "data"), "FPH_EXPORT_DIR": str(tmp / "exports"),
                       "FPH_WAREHOUSE": str(tmp / "warehouse.duckdb"), "PIA_WAREHOUSE": str(tmp / "warehouse.duckdb")})
    fph = pytest.importorskip("fph.pipeline", reason="install field-performance-hub to build test data")
    fph.run(verbose=False)
    return tmp / "warehouse.duckdb"
