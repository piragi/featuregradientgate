# Lazy re-export: run_unified_pipeline still lives in root pipeline.py
# (not yet migrated to a package module). Eager import causes a circular
# dependency because pipeline.py imports from gradcamfaith.models.load,
# which triggers this __init__.py, which would try to import from
# pipeline.py again before it has finished loading.
def __getattr__(name):
    if name == "run_unified_pipeline":
        from pipeline import run_unified_pipeline
        return run_unified_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
