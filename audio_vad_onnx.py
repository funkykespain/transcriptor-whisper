# -*- coding: utf-8 -*-
"""Backend VAD 100 % ONNX de Silero (sin PyTorch) para entornos ligeros/ARM.

Motivación: en servidores ARM (p. ej. Ampere A1) o imágenes Docker sin GPU,
descargar PyTorch desde PyPI (con librerías CUDA) infla la imagen >2 GB y no
aporta nada. El modelo `silero_vad.onnx` (opset 16) viene **empaquetado** en
el wheel de ``silero-vad`` y solo necesita ``onnxruntime`` + ``numpy`` para
inferir.

Este módulo:
  * localiza ``silero_vad.onnx`` **sin importar** el paquete (``find_spec``),
    de modo que torch jamás se carga en memoria;
  * replica el bucle ventana (512) + contexto (64) + estado (2x1x128) de la
    clase ``OnnxWrapper`` del paquete oficial;
  * implementa la post-proceso clásico de ``get_speech_timestamps``
    (histéresis threshold/neg_threshold, duración mínima de habla/silencio,
    cortes de tramos largos) y delega el padding de 400 ms en
    :func:`audio_vad._pad_and_merge`.

Configuración:
  * ``SILERO_VAD_ONNX_PATH``: ruta alternativa del modelo .onnx.
  * Sin red y sin paquete instalado, se intenta descargar el modelo a
    ``~/.cache/silero_vad/`` la primera vez.
"""

from __future__ import annotations

import importlib.util
import math
import os
import urllib.request
from typing import List, Sequence, Tuple

import numpy as np

from audio_vad import SpeechSegment, VadError, _pad_and_merge, _read_pcm_samples

#: Ventana de inferencia a 16 kHz (512 muestras = 32 ms).
_WINDOW_SAMPLES: int = 512
#: Contexto que se antepone a cada ventana (64 muestras).
_CONTEXT_SAMPLES: int = 64
#: Grosor del estado recurrente del modelo (2, 1, 128).
_STATE_SHAPE: Tuple[int, int, int] = (2, 1, 128)
#: URLs de respaldo para descargar el modelo si no viene empaquetado.
_DEFAULT_ONNX_URLS: Tuple[str, ...] = (
    "https://models.silero.ai/vad_models/onnx/silero_vad.onnx",
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/files/silero_vad.onnx",
)


# ---------------------------------------------------------------------------
# Resolución del modelo
# ---------------------------------------------------------------------------
def resolve_onnx_model_path() -> str:
    """Devuelve la ruta de ``silero_vad.onnx`` sin importar el paquete silero-vad.

    Orden: variable de entorno ``SILERO_VAD_ONNX_PATH`` -> fichero empaquetado
    en ``silero_vad/data/`` (``find_spec`` no ejecuta el paquete, por lo que
    torch no se carga) -> descarga a ``~/.cache/silero_vad/``.
    """
    env = os.environ.get("SILERO_VAD_ONNX_PATH", "").strip()
    if env:
        if not os.path.isfile(env):
            raise VadError(f"SILERO_VAD_ONNX_PATH apunta a un archivo inexistente: {env}")
        return env

    try:
        spec = importlib.util.find_spec("silero_vad")
        if spec is not None and spec.origin:
            candidate = os.path.join(
                os.path.dirname(spec.origin), "data", "silero_vad.onnx"
            )
            if os.path.isfile(candidate):
                return candidate
    except (ImportError, ValueError):
        pass

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "silero_vad")
    cached = os.path.join(cache_dir, "silero_vad.onnx")
    if os.path.isfile(cached):
        return cached

    os.makedirs(cache_dir, exist_ok=True)
    last_error: Exception | None = None
    for url in _DEFAULT_ONNX_URLS:
        try:
            tmp = f"{cached}.part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, cached)
            return cached
        except Exception as exc:  # noqa: BLE001 - probamos el siguiente espejo
            last_error = exc
    raise VadError(f"No se pudo obtener silero_vad.onnx: {last_error}")


# ---------------------------------------------------------------------------
# Predictor ventana a ventana (equivale a OnnxWrapper)
# ---------------------------------------------------------------------------
class SileroVadONNX:
    """Inferencia de voz por ventanas de 512 muestras con estado recurrente.

    Mismo contrato que el ``OnnxWrapper`` del paquete oficial: la entrada de
    cada llamada es la ventana actual (512 muestras) y la clase mantiene el
    contexto (64) y el estado (2, 1, 128) entre llamadas.
    """

    def __init__(self, session) -> None:
        self._session = session
        self.reset_states()

    def reset_states(self) -> None:
        """Reinicia contexto y estado (llamar por cada audio nuevo)."""
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, 0), dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        """Devuelve la probabilidad de voz del tramo de 32 ms dado."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(1, -1)
        if chunk.shape[1] != _WINDOW_SAMPLES:
            raise ValueError(
                f"SileroVadONNX espera ventanas de {_WINDOW_SAMPLES} muestras, "
                f"se recibieron {chunk.shape[1]}."
            )
        if self._context.shape[1] == 0:
            self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)
        x = np.concatenate([self._context, chunk], axis=1)  # (1, 64+512=576)
        ort_inputs = {
            "input": x,
            "state": self._state,
            "sr": np.array(16000, dtype=np.int64),
        }
        out, state = self._session.run(None, ort_inputs)
        self._state = state
        self._context = x[:, -_CONTEXT_SAMPLES:]
        return float(np.asarray(out).reshape(-1)[0])


# ---------------------------------------------------------------------------
# Sesión ONNX en caché
# ---------------------------------------------------------------------------
_onnx_session = None
_onnx_session_path = None


def get_onnx_session():
    """Carga (una sola vez) la InferenceSession de ``silero_vad.onnx``."""
    global _onnx_session, _onnx_session_path
    path = resolve_onnx_model_path()
    if _onnx_session is None or _onnx_session_path != path:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise VadError(
                "El backend ONNX necesita 'onnxruntime' (pip install onnxruntime)."
            ) from exc
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        _onnx_session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        _onnx_session_path = path
    return _onnx_session


# ---------------------------------------------------------------------------
# Post-proceso: probabilidades por ventana -> tramos de voz (ms)
# ---------------------------------------------------------------------------
def probs_to_speech_segments_ms(
    probs: Sequence[float],
    *,
    window_ms: float,
    threshold: float = 0.5,
    neg_threshold: float = 0.35,
    min_speech_duration_ms: float = 250.0,
    min_silence_duration_ms: float = 100.0,
    max_speech_duration_s: float = float("inf"),
) -> List[Tuple[float, float]]:
    """Convierte probabilidades por ventana en tramos (start_ms, end_ms).

    Réplica del algoritmo de ``get_speech_timestamps`` de silero-vad:
    histéresis (entra con ``threshold``, sale tras ``min_silence_duration_ms``
    por debajo de ``neg_threshold``), filtro de duración mínima y división de
    tramos que superan ``max_speech_duration_s``.
    """
    if not probs:
        return []
    if min_speech_duration_ms < 0 or min_silence_duration_ms < 0:
        raise ValueError("Las duraciones mínimas no pueden ser negativas")

    speech_frames = max(1, int(round(min_silence_duration_ms / window_ms)))
    segments: List[Tuple[float, float]] = []
    triggered = False
    speech_start_ms = 0.0
    silence_frames = 0
    total_ms = len(probs) * window_ms

    for index, prob in enumerate(probs):
        end_ms = (index + 1) * window_ms
        if not triggered:
            if prob >= threshold:
                triggered = True
                speech_start_ms = index * window_ms
                silence_frames = 0
        elif prob >= neg_threshold:
            silence_frames = 0
        else:
            silence_frames += 1
            if silence_frames >= speech_frames:
                segments.append((speech_start_ms, end_ms))
                triggered = False
                silence_frames = 0

    if triggered:
        segments.append((speech_start_ms, total_ms))

    segments = [s for s in segments if (s[1] - s[0]) >= min_speech_duration_ms]

    if not math.isinf(max_speech_duration_s):
        max_ms = max_speech_duration_s * 1000.0
        split: List[Tuple[float, float]] = []
        for start, end in segments:
            while (end - start) > max_ms:
                split.append((start, start + max_ms))
                start += max_ms
            split.append((start, end))
        segments = split
    return segments


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def detect_speech_segments_onnx(
    input_path,
    *,
    sampling_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 400,
    max_speech_duration_s: float = float("inf"),
) -> List[SpeechSegment]:
    """Detecta voz con el modelo ONNX puro (sin torch) y aplica el padding.

    Misma interfaz y semántica que
    :func:`audio_vad.detect_speech_segments` (entrada normalizada de la
    Fase 1, padding de ``speech_pad_ms`` antes y después, fusión de solapes
    y recorte en los bordes), pero usando ``onnxruntime`` + ``numpy``.

    Returns:
        Lista de :class:`SpeechSegment` con los timestamps en segundos.
    """
    samples = _read_pcm_samples(input_path, sampling_rate)
    duration_s = samples.size / sampling_rate
    if samples.size < _WINDOW_SAMPLES:
        return []  # Audio más corto que la ventana mínima de inferencia.

    session = get_onnx_session()
    predictor = SileroVadONNX(session)
    window_ms = 1000.0 * _WINDOW_SAMPLES / sampling_rate  # 32 ms a 16 kHz

    probs: List[float] = []
    for start in range(0, samples.size - _WINDOW_SAMPLES + 1, _WINDOW_SAMPLES):
        chunk = samples[start:start + _WINDOW_SAMPLES].astype(np.float32) / 32768.0
        probs.append(predictor(chunk))

    raw_ms = probs_to_speech_segments_ms(
        probs,
        window_ms=window_ms,
        threshold=threshold,
        neg_threshold=threshold - 0.15,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
    )
    raw_seconds = [(start / 1000.0, end / 1000.0) for start, end in raw_ms]
    padded = _pad_and_merge(raw_seconds, duration_s, speech_pad_ms)
    return [SpeechSegment(start=start, end=end) for start, end in padded]