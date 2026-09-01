from .build import EVALUATOR_REGISTRY, build_evaluator

__all__ = ['EVALUATOR_REGISTRY', 'build_evaluator', 'VLNCEEvaluator']


def __getattr__(name):
    if name == 'VLNCEEvaluator':
        from .vlnce_eval import VLNCEEvaluator

        return VLNCEEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
