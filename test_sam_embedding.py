import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sam_embedding import write_artifacts


class WriteArtifactsTest(unittest.TestCase):
    def test_writes_greenskeye_manifest_and_raw_embedding(self) -> None:
        embedding = np.arange(24, dtype=np.float64).reshape(1, 2, 3, 4)

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "output"
            manifest_path, embedding_path = write_artifacts(
                output_dir,
                embedding,
                model_type="vit_h",
                checkpoint="sam_vit_h.pth",
                original_size=(600, 800),
                input_size=(768, 1024),
            )

            self.assertEqual(
                embedding_path.read_bytes(), embedding.astype("<f4").tobytes()
            )
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                {
                    "imageSize": {"width": 800, "height": 600},
                    "embedding": "embedding.bin",
                    "embeddingShape": [1, 2, 3, 4],
                    "embeddingDtype": "float32",
                    "modelType": "vit_h",
                    "checkpoint": "sam_vit_h.pth",
                    "inputSize": {"width": 1024, "height": 768},
                },
            )


if __name__ == "__main__":
    unittest.main()
