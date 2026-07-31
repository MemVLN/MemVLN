import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "trainer" / "leo_trainer.py"
TRAINER_SOURCE = TRAINER_PATH.read_text(encoding="utf-8")
TRAINER_TREE = ast.parse(TRAINER_SOURCE, filename=str(TRAINER_PATH))
TRAINER_BUILD_PATH = ROOT / "trainer" / "build.py"
TRAINER_BUILD_SOURCE = TRAINER_BUILD_PATH.read_text(encoding="utf-8")
TRAINER_BUILD_TREE = ast.parse(
    TRAINER_BUILD_SOURCE,
    filename=str(TRAINER_BUILD_PATH),
)


def _load_report_helper():
    selected_nodes = []
    for node in TRAINER_TREE.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_DEPRECATED_CHECKPOINT_KEY_COMPONENTS"
                for target in node.targets
            )
        ):
            selected_nodes.append(node)
        elif (
            isinstance(node, ast.FunctionDef)
            and node.name == "report_model_load_incompatibilities"
        ):
            selected_nodes.append(node)

    namespace = {"logger": Mock()}
    helper_module = ast.Module(body=selected_nodes, type_ignores=[])
    exec(compile(helper_module, str(TRAINER_PATH), "exec"), namespace)
    return namespace["report_model_load_incompatibilities"]


report_model_load_incompatibilities = _load_report_helper()


def _load_checkpoint_path_helpers():
    selected_nodes = []
    selected_names = {
        "resolve_checkpoint_total_limit",
        "resolve_model_checkpoint_path",
    }
    for node in TRAINER_TREE.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_MODEL_CHECKPOINT_FILENAMES"
                for target in node.targets
            )
        ):
            selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in selected_names:
            selected_nodes.append(node)

    namespace = {"Path": Path, "latest_checkpoint": latest_checkpoint}
    helper_module = ast.Module(body=selected_nodes, type_ignores=[])
    exec(compile(helper_module, str(TRAINER_PATH), "exec"), namespace)
    return (
        namespace["resolve_checkpoint_total_limit"],
        namespace["resolve_model_checkpoint_path"],
    )

def _load_checkpoint_index_helpers():
    selected_names = {"checkpoint_index", "latest_checkpoint"}
    selected_nodes = []
    for node in TRAINER_BUILD_TREE.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_CHECKPOINT_DIRECTORY_PATTERN"
                for target in node.targets
            )
        ):
            selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in selected_names:
            selected_nodes.append(node)

    namespace = {"Path": Path, "os": __import__("os"), "re": __import__("re")}
    helper_module = ast.Module(body=selected_nodes, type_ignores=[])
    exec(compile(helper_module, str(TRAINER_BUILD_PATH), "exec"), namespace)
    return namespace["checkpoint_index"], namespace["latest_checkpoint"]


checkpoint_index, latest_checkpoint = _load_checkpoint_index_helpers()
resolve_checkpoint_total_limit, resolve_model_checkpoint_path = (
    _load_checkpoint_path_helpers()
)


class CheckpointCompatibilityReportTests(unittest.TestCase):
    def test_deprecated_keys_are_separated_without_hiding_other_mismatches(self):
        log = Mock()

        report = report_model_load_incompatibilities(
            ["current_head.weight"],
            [
                "memory_repa_adapter.proj.weight",
                "module.model.model.memory_repa_adapter.proj.bias",
                "misspelled_module.weight",
                "model.memory_repa_adapter_backup.weight",
            ],
            log=log,
        )

        self.assertEqual(report["missing_keys"], ("current_head.weight",))
        self.assertEqual(
            report["deprecated_unexpected_keys"],
            (
                "memory_repa_adapter.proj.weight",
                "module.model.model.memory_repa_adapter.proj.bias",
            ),
        )
        self.assertEqual(
            report["unexpected_keys"],
            (
                "misspelled_module.weight",
                "model.memory_repa_adapter_backup.weight",
            ),
        )
        warnings = "\n".join(call.args[0] for call in log.warning.call_args_list)
        self.assertIn("Deprecated unexpected checkpoint keys", warnings)
        self.assertIn("Missing model checkpoint keys", warnings)
        self.assertIn("Unexpected model checkpoint keys", warnings)
        self.assertIn("misspelled_module.weight", warnings)
        self.assertIn("memory_repa_adapter_backup.weight", warnings)

    def test_only_removed_component_keys_are_explicitly_deprecated(self):
        log = Mock()

        report = report_model_load_incompatibilities(
            [],
            ["model.model.memory_repa_adapter.proj.weight"],
            log=log,
        )

        self.assertEqual(report["unexpected_keys"], ())
        self.assertEqual(
            report["deprecated_unexpected_keys"],
            ("model.model.memory_repa_adapter.proj.weight",),
        )
        log.warning.assert_called_once()
        log.info.assert_called_once_with(
            "No active model checkpoint key mismatches."
        )

    def test_model_only_load_has_one_common_reporter_after_load_branches(self):
        trainer_class = next(
            node
            for node in TRAINER_TREE.body
            if isinstance(node, ast.ClassDef) and node.name == "LeoTrainer"
        )
        load_method = next(
            node
            for node in trainer_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "load"
        )
        model_only_branch = next(
            node
            for node in load_method.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "model_only"
        )
        reporter_calls = [
            node
            for node in ast.walk(model_only_branch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "report_model_load_incompatibilities"
        ]

        self.assertEqual(len(reporter_calls), 1)
        self.assertEqual(
            [argument.id for argument in reporter_calls[0].args],
            ["missing_keys", "unexpected_keys"],
        )


class CheckpointRuntimeContractTests(unittest.TestCase):
    def test_latest_checkpoint_uses_full_numeric_suffix_not_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_9 = root / "checkpoint_9"
            checkpoint_10 = root / "checkpoint_10"
            checkpoint_33 = root / "checkpoint_33"
            for checkpoint in (checkpoint_9, checkpoint_10, checkpoint_33):
                checkpoint.mkdir()
            (root / "notes.txt").write_text("not a checkpoint", encoding="utf-8")
            (root / "checkpoint_backup").mkdir()

            # Make the numerically oldest checkpoint the newest by mtime.
            os.utime(checkpoint_9, (300, 300))
            os.utime(checkpoint_10, (200, 200))
            os.utime(checkpoint_33, (100, 100))

            self.assertEqual(latest_checkpoint(root), str(checkpoint_33))
            self.assertEqual(checkpoint_index(f"{checkpoint_33}/"), 33)

    def test_checkpoint_index_rejects_malformed_directory_names(self):
        for path in ("checkpoint_latest", "run_33", "checkpoint_-1"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "checkpoint_<number>"):
                    checkpoint_index(path)

    def test_model_checkpoint_directory_resolves_supported_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            safetensors_path = root / "model.safetensors"
            safetensors_path.touch()

            self.assertEqual(
                resolve_model_checkpoint_path(root),
                safetensors_path,
            )
            self.assertEqual(
                resolve_model_checkpoint_path(safetensors_path),
                safetensors_path,
            )

    def test_model_checkpoint_directory_fails_on_none_or_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "No model checkpoint"):
                resolve_model_checkpoint_path(root)

            (root / "model.safetensors").touch()
            (root / "pytorch_model.bin").touch()
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                resolve_model_checkpoint_path(root)

    def test_experiment_root_resolves_latest_numeric_model_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_9 = root / "checkpoints" / "checkpoint_9"
            checkpoint_33 = root / "checkpoints" / "checkpoint_33"
            checkpoint_9.mkdir(parents=True)
            checkpoint_33.mkdir()
            old_model = checkpoint_9 / "model.safetensors"
            latest_model = checkpoint_33 / "model.safetensors"
            old_model.touch()
            latest_model.touch()

            # Reverse mtimes to ensure numeric selection remains authoritative.
            os.utime(checkpoint_9, (300, 300))
            os.utime(checkpoint_33, (100, 100))

            self.assertEqual(resolve_model_checkpoint_path(root), latest_model)
            self.assertEqual(
                resolve_model_checkpoint_path(root / "checkpoints"),
                latest_model,
            )

    def test_checkpoint_total_limit_accepts_positive_integer_or_null(self):
        self.assertEqual(
            resolve_checkpoint_total_limit(
                SimpleNamespace(get=lambda key, default: 5)
            ),
            5,
        )
        self.assertIsNone(
            resolve_checkpoint_total_limit(
                SimpleNamespace(get=lambda key, default: None)
            )
        )

        for value, error in ((0, ValueError), (-1, ValueError), (True, TypeError)):
            with self.subTest(value=value):
                with self.assertRaises(error):
                    resolve_checkpoint_total_limit(
                        SimpleNamespace(get=lambda key, default, value=value: value)
                    )

    def test_periodic_hooks_run_after_update_and_validation_receives_epoch(self):
        trainer_class = next(
            node
            for node in TRAINER_TREE.body
            if isinstance(node, ast.ClassDef) and node.name == "LeoTrainer"
        )
        train_step = next(
            node
            for node in trainer_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_step"
        )
        step_loader_call = next(
            node
            for node in ast.walk(train_step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step_loader"
        )
        validation_call = next(
            node
            for node in ast.walk(train_step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "val_step"
        )
        save_calls = [
            node
            for node in ast.walk(train_step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "save"
        ]

        self.assertEqual(
            [argument.id for argument in validation_call.args],
            ["epoch"],
        )
        self.assertGreater(validation_call.lineno, step_loader_call.lineno)
        self.assertTrue(save_calls)
        self.assertTrue(
            all(call.lineno > step_loader_call.lineno for call in save_calls)
        )


if __name__ == "__main__":
    unittest.main()
