from importlib import import_module

from fvcore.common.registry import Registry

EVALUATOR_REGISTRY = Registry('Evaluator')

_EVALUATOR_MODULES = {
    'VLNCEEvaluator': 'evaluator.vlnce_eval',
}


def build_eval_leo(cfg, dataset_name, evaluator_name):
    module_name = _EVALUATOR_MODULES.get(evaluator_name)
    if module_name is not None:
        import_module(module_name)
    return EVALUATOR_REGISTRY.get(evaluator_name)(cfg, dataset_name)
