"""Regression checks for notebook evaluation, without model downloads or training."""
import ast
import json
from pathlib import Path
import unittest


NOTEBOOK = Path(__file__).resolve().parents[1] / "Caption Generation Project.ipynb"


def score_assignment():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assignments = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or "rouge_score_value =" not in source:
            continue
        for statement in ast.parse(source).body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "rouge_score_value"
                for target in statement.targets
            ):
                assignments.append(statement)
    if len(assignments) != 1:
        raise AssertionError("Expected exactly 1 notebook ROUGE score assignment")
    return compile(ast.Module(body=assignments, type_ignores=[]), "caption-evaluation-cell", "exec")


class AlignedScore:
    """Small metric double that requires the full aligned evaluation batch."""

    def __call__(self, predictions, references):
        if not isinstance(references, list) or len(predictions) != len(references):
            raise ValueError("Predictions and references must be aligned lists")
        matches = sum(prediction == reference for prediction, reference in zip(predictions, references))
        return {"rougeL_fmeasure": matches / len(predictions)}


class NotebookEvaluationTests(unittest.TestCase):
    def test_every_prediction_receives_its_own_reference(self):
        namespace = {
            "predictions": ["a red car", "a sleeping cat"],
            "pred_results": {"references": ["a red car", "a running dog"]},
            "references": "a running dog",  # Value left by the final generation iteration.
            "rouge_score": AlignedScore(),
        }
        exec(score_assignment(), namespace)
        self.assertEqual(namespace["rouge_score_value"], 0.5)

    def test_scoring_does_not_depend_on_a_stale_loop_variable(self):
        namespace = {
            "predictions": ["a red car"],
            "pred_results": {"references": ["a red car"]},
            "rouge_score": AlignedScore(),
        }
        exec(score_assignment(), namespace)
        self.assertEqual(namespace["rouge_score_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
