import importlib
import sys
import unittest
from unittest import mock


class EvaluatorLazyImportTest(unittest.TestCase):
    def test_import_build_does_not_import_habitat(self):
        before = set(sys.modules)

        build = importlib.import_module('evaluator.build')

        imported = set(sys.modules) - before
        self.assertNotIn('habitat', imported)
        self.assertFalse(any(name.startswith('habitat.') for name in imported))
        self.assertNotIn('habitat_baselines', imported)
        self.assertFalse(any(name.startswith('habitat_baselines.') for name in imported))
        self.assertNotIn('evaluator.vlnce_eval', imported)
        self.assertTrue(callable(build.build_eval_leo))

    def test_vlnce_is_loaded_only_when_requested(self):
        build = importlib.import_module('evaluator.build')

        with mock.patch.object(build, 'import_module') as lazy_import:
            with self.assertRaises(KeyError):
                build.build_eval_leo(None, 'mp3d_r2r', 'VLNCEEvaluator')

        lazy_import.assert_called_once_with('evaluator.vlnce_eval')

    def test_unknown_evaluator_keeps_registry_error(self):
        build = importlib.import_module('evaluator.build')

        with mock.patch.object(build, 'import_module') as lazy_import:
            with self.assertRaises(KeyError):
                build.build_eval_leo(None, 'mp3d_r2r', 'UnknownEvaluator')

        lazy_import.assert_not_called()


if __name__ == '__main__':
    unittest.main()
