"""Exercise token/mask contracts with small doubles, without downloading PyTorch."""
import ast
import json
import os
from pathlib import Path
import random
import tempfile
import types
import unittest

import numpy as np
import pandas as pd
from PIL import Image


def definitions():
    path = Path(__file__).resolve().parents[1] / "Caption Generation Project.ipynb"
    namespace = {"Dataset": object, "pd": pd, "Path": Path, "os": os,
                 "Image": Image, "random": random,
                 "torch": types.SimpleNamespace(
                     tensor=np.asarray, bool=np.bool_,
                     ones=lambda shape, dtype, device: np.ones(shape, dtype=dtype),
                     triu=lambda tensor, diagonal=0: np.triu(tensor, k=diagonal))}
    for cell in json.loads(path.read_text())["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or source.lstrip().startswith("!"):
            continue
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in {
                     "CustomImageDataset", "create_mask", "generate_square_subsequent_mask"}]
        if nodes:
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "<notebook>", "exec"), namespace)
    return namespace


class Tokenizer:
    bos_token_id, eos_token_id, pad_token_id = 101, 102, 0

    def encode(self, text, add_special_tokens=False):
        return [int(token) for token in text.split()]

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(token) for token in tokens if token not in {0, 101, 102})


class CaptionDataTests(unittest.TestCase):
    def setUp(self):
        self.ns = definitions()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        Image.new("RGB", (2, 2)).save(self.root / "sample.jpg")
        (self.root / "split.txt").write_text("sample.jpg\n")

    def dataset(self, captions, **kwargs):
        return self.ns["CustomImageDataset"](
            self.root, {"sample.jpg": captions}, self.root / "split.txt",
            tokenizer=Tokenizer(), max_len=5, **kwargs)

    def test_long_and_short_captions_form_a_fixed_size_batch_and_keep_eos(self):
        long = self.dataset(["1 2 3 4 5 6 7"])[0]["caption"]
        short = self.dataset(["8"])[0]["caption"]
        np.testing.assert_array_equal(long, [101, 1, 2, 3, 4, 102])
        np.testing.assert_array_equal(short, [101, 8, 102, 0, 0, 0])
        self.assertEqual(np.stack([long, short]).shape, (2, 6))

    def test_evaluation_uses_a_stable_reference(self):
        dataset = self.dataset(["1 2", "8 9"], phase="eval")
        for _ in range(10):
            np.testing.assert_array_equal(dataset[0]["caption"], [101, 1, 2, 102, 0, 0])

    def test_causal_mask_blocks_only_future_positions(self):
        mask = self.ns["generate_square_subsequent_mask"](3)
        self.assertEqual(mask.dtype, np.bool_)
        np.testing.assert_array_equal(mask, [[False, True, True], [False, False, True], [False, False, False]])

    def test_padding_mask_stays_boolean_instead_of_becoming_positive_attention_bias(self):
        class Tokens(np.ndarray):
            @property
            def device(self):
                return "cpu"
        tokens = np.asarray([[101, 7, 0], [101, 0, 0]]).view(Tokens)
        causal, padding = self.ns["create_mask"](None, tokens, pad_idx=0)
        self.assertEqual(causal.dtype, padding.dtype)
        np.testing.assert_array_equal(padding, [[False, False, True], [False, True, True]])


if __name__ == "__main__":
    unittest.main()
