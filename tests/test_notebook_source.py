"""Static checks on the notebook source; nothing is executed or downloaded."""
import ast
import json
from pathlib import Path
import unittest

NOTEBOOK = Path(__file__).resolve().parents[1] / "Caption Generation Project.ipynb"


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        # Shell (!) and magic (%) lines are IPython syntax, not Python.
        cells.append("\n".join("" if line.lstrip().startswith(("!", "%")) else line for line in lines))
    return cells


def call_keywords(source, function_name):
    """Keyword arguments of every call to `function_name` in `source`."""
    calls = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name == function_name:
                calls.append({keyword.arg: keyword.value for keyword in node.keywords})
    return calls


class NotebookSourceTests(unittest.TestCase):
    def setUp(self):
        self.cells = code_cells()
        self.source = "\n".join(self.cells)

    def test_every_code_cell_parses(self):
        for index, cell in enumerate(self.cells):
            with self.subTest(code_cell=index):
                compile(cell, f"code-cell-{index}", "exec")

    def test_cnn_optimizers_use_the_cnn_learning_rate(self):
        cnn_optimizers = [
            node for cell in self.cells for node in ast.walk(ast.parse(cell))
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "AdamW"
            and node.args and "image_encoder" in ast.unparse(node.args[0])
        ]
        self.assertEqual(len(cnn_optimizers), 2)
        for node in cnn_optimizers:
            learning_rate = {keyword.arg: ast.unparse(keyword.value) for keyword in node.keywords}["lr"]
            self.assertEqual(learning_rate, "CNN_LEARNING_RATE")

    def test_generation_uses_a_causal_mask(self):
        tree = ast.parse(next(cell for cell in self.cells if "def generate(" in cell))
        generate = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "generate")
        decoder_calls = [node for node in ast.walk(generate) if isinstance(node, ast.Call)
                         and ast.unparse(node.func) == "self.text_decoder"]
        self.assertEqual(len(decoder_calls), 1)
        self.assertIn("tgt_mask", {keyword.arg for keyword in decoder_calls[0].keywords})

    def test_checkpoint_is_loaded_portably(self):
        loads = [keywords for cell in self.cells for keywords in call_keywords(cell, "load")
                 if "torch.load" in cell]
        self.assertTrue(loads)
        self.assertTrue(all("map_location" in keywords for keywords in loads))
        self.assertNotIn('PATH = "/kaggle/working', self.source)


if __name__ == "__main__":
    unittest.main()
