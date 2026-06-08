try:
    from gliner import GLiNER
except Exception:
    GLiNER = None


_model = None


def _get_model():
    global _model
    if _model is None:
        if GLiNER is None:
            raise ImportError("GLiNER is not installed")
        _model = GLiNER.from_pretrained("urchade/gliner_base")
    return _model


def extract_entities(text):
    if not text or not str(text).strip():
        return []

    model = _get_model()
    entities = model.predict_entities(
        str(text),
        ["research topic", "method", "domain"],
    )
    return [entity["text"] for entity in entities if entity.get("text")]
