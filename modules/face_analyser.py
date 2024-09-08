from typing import Any
import insightface

import modules.globals
from modules.typing import Frame

FACE_ANALYSER = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        print(f"Using execution providers: {modules.globals.execution_providers}")  # Add this line for debugging
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=modules.globals.execution_providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
        print(f"Models: {FACE_ANALYSER.models}")  # Add this line for debugging
        for model_name, model in FACE_ANALYSER.models.items():
            print(f"Model {model_name} using providers: {model.session.get_providers()}")  # Add this line for debugging
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None
