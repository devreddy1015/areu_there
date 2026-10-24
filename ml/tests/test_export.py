"""ml/export/onnx_export.py (Phase 12): a model's ONNX export equals PyTorch
within 1e-4 on batches with 1 to 132 links, padding and unknown geometry."""

import os
import shutil
import tempfile
import unittest

try:
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
    HAVE_ORT = True
except ImportError:  # pragma: no cover
    HAVE_ORT = False

from ml.train import model as mdl


@unittest.skipUnless(HAVE_ORT, "onnx and onnxruntime are needed")
class TestOnnxExport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="onnx-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_matches_pytorch(self):
        from ml.export import onnx_export as ox
        for n_ref, t in ((44, 10), (40, 20)):
            m = mdl.build_model(mdl.ModelConfig(n_ref=n_ref, d=64), seed=1).eval()
            path = os.path.join(self.dir, f"m{n_ref}.onnx")
            info = ox.export(m, path, t)
            self.assertEqual((info["t"], info["n_ref"]), (t, n_ref))
            batches = [ox.random_batch(n_ref, t, b=3, n_links=n, seed=s) for s, n in enumerate((1, 7, 56, 132))]
            res = ox.check(m, path, batches)
            self.assertTrue(res["ok"], res)
            self.assertLess(res["max"], ox.TOLERANCE)
            self.assertEqual(res["windows"], 12)


if __name__ == "__main__":
    unittest.main()
