import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_test_dir = Path(__file__).resolve().parent
_import_paths = []
for _parent in _test_dir.parents:
    _import_paths.extend((_parent / "src", _parent))
_import_paths.extend((_test_dir / "src", _test_dir))
for _path in _import_paths:
    if (_path / "carbon_audit").is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

if importlib.util.find_spec("sklearn") is None:
    raise unittest.SkipTest("scikit-learn is not installed")

from carbon_audit import Workload, phases
_model = importlib.import_module("carbon_audit.model")
synthetic = _model.synthetic
train = _model.train
CarbonModel = _model.CarbonModel


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        path = cls.temp_dir.name + "/m.joblib"
        rows, y = synthetic(1500)
        cls.metrics = train(rows, y, path)
        cls.model = CarbonModel(path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_accuracy(self):
        self.assertLess(self.metrics["mape"], 0.25)

    def test_prediction_close_to_physics(self):
        w = Workload()
        p = self.model.predict(w)
        expected = sum(phases(w).values())
        self.assertLess(abs(p["kwh"] - expected) / expected, 0.35)

    def test_alert_levels(self):
        w = Workload()
        self.assertEqual(self.model.alert(w, 1000)["level"], "ok")
        self.assertEqual(self.model.alert(w, 0.001)["level"], "critical")
