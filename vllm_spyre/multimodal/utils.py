"""
Super hacky utils for multimodal model stuff.
"""
def is_multimodal(fms_model, fms_mm_registry):
    if fms_mm_registry is None:
        return False
    for mm_type in fms_mm_registry:
        if isinstance(fms_model, mm_type):
            return True
    return False
