# -*- coding: utf-8 -*-
"""Fase 2 del pipeline ASR: detección de voz (VAD) sobre audio ya normalizado.

Entrada esperada: el WAV PCM producido por
:func:`audio_preprocessing.preprocess_audio_base` (16 kHz / mono / 16-bit).

Flujo:
  1. Se cargan los timestamps crudos de voz con **Silero VAD** (API oficial v6:
     ``load_silero_vad()`` + ``get_speech_timestamps()``; documentación
     consultada en context7 -> /snakers4/silero-vad).
  2. Se aplica un **padding de 400 ms antes y después** de cada fragmento
     (``speech_pad_ms``), recortando en los bordes del audio y fusionando los
     fragmentos que se solapan tras el padding.
  3. Se devuelve una lista estructurada de :class:`SpeechSegment` (timestamps
     en segundos y ms) y se puede exportar cada chunk a un WAV independiente.

Nota: el padding se aplica aquí de forma explícita (no vía
``speech_pad_ms`` de ``get_speech_timestamps``, que reparte a partes iguales
a ambos lados) para cumplir exactamente el requisito de 400 ms por lado.
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

DEFAULT_SAMPLE_RATE: int = 16_000              # Tasa del audio normalizado (Fase 1)
DEFAULT_THRESHOLD: float = 0.5                 # Umbral de probabilidad de voz
DEFAULT_MIN_SPEECH_DURATION_MS: int = 250      # Ignora ráfagas < 250 ms
DEFAULT_MIN_SILENCE_DURATION_MS: int = 100     # Cierra el tramo tras 100 ms de silencio
DEFAULT_SPEECH_PAD_MS: int = 400               # Padding antes y después de cada tramo
_VAD_WINDOW_SAMPLES: int = 512                 # Ventana mínima Silero a 16 kHz

PathLike = Union[str, os.PathLike]


class VadError(RuntimeError):
    """Error controlado del pipeline VAD (formato inválido, modelo ausente...)."""


@dataclass(frozen=True)
class SpeechSegment:
    """Tramo de voz detectado, en segundos (puede incluir el padding)."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(
                f"Segmento inválido: start={self.start!r}, end={self.end!r}"
            )

    @property
    def duration(self) -> float:
        """Duración del tramo en segundos."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, float]:
        """Estructura serializable con timestamps en segundos y milisegundos."""
        return {
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "duration": round(self.duration, 4),
            "start_ms": int(round(self.start * 1000)),
            "end_ms": int(round(self.end * 1000)),
        }

    def to_ms(self) -> Tuple[int, int]:
        """Devuelve (inicio_ms, fin_ms) para integrar con pydub etc."""
        return self.to_dict()["start_ms"], self.to_dict()["end_ms"]


# ---------------------------------------------------------------------------
# Carga perezosa y en caché del modelo Silero VAD
# ---------------------------------------------------------------------------
_vad_model = None


def get_vad_model():
    """Carga (una sola vez) el modelo Silero VAD v6.

    La carga se delega en ``silero_vad.load_silero_vad``, que descarga el
    modelo JIT la primera vez y lo cachea en el hub de torch. Por defecto usa
    el runtime de PyTorch; puede forzarse ONNX con ``onnx=True``.
    """
    global _vad_model
    if _vad_model is None:
        try:
            from silero_vad import load_silero_vad
        except ModuleNotFoundError as exc:
            raise VadError(
                "La dependencia 'silero-vad' no está instalada. "
                "Ejecuta: pip install silero-vad"
            ) from exc
        _vad_model = load_silero_vad()
    return _vad_model


# ---------------------------------------------------------------------------
# Lectura / validación del audio normalizado (WAV PCM 16 kHz mono 16-bit)
# ---------------------------------------------------------------------------
def _read_pcm_samples(path: PathLike, sampling_rate: int) -> np.ndarray:
    """Lee y valida un WAV PCM; devuelve los samples int16 como np.ndarray 1-D."""
    if not os.path.isfile(path):
        raise VadError(f"El archivo de entrada no existe: {path}")
    try:
        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
    except (wave.Error, EOFError) as exc:
        raise VadError(f"'{path}' no es un archivo WAV válido: {exc}") from exc

    if sample_rate != sampling_rate:
        raise VadError(
            f"'{path}' no está normalizado a {sampling_rate} Hz (tiene {sample_rate} Hz). "
            "Ejecuta primero audio_preprocessing.preprocess_audio_base (Fase 1)."
        )
    if channels != 1:
        raise VadError(
            f"'{path}' no está normalizado a mono (tiene {channels} canales). "
            "Ejecuta primero audio_preprocessing.preprocess_audio_base (Fase 1)."
        )
    if sampwidth != 2:  # 16-bit PCM
        raise VadError(
            f"'{path}' no es PCM 16-bit (sampwidth={sampwidth}). "
            "Ejecuta primero audio_preprocessing.preprocess_audio_base (Fase 1)."
        )

    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(n_frames)
    return np.frombuffer(raw, dtype="<i2").astype(np.int16)


def read_normalized_audio(input_path: PathLike, sampling_rate: int = DEFAULT_SAMPLE_RATE
                          ) -> "torch.Tensor":
    """Devuelve el audio normalizado como tensor float32 mono (1-D), lista para VAD."""
    import torch  # importe perezoso: el módulo es importable sin torch instalado

    samples = _read_pcm_samples(input_path, sampling_rate)
    return torch.from_numpy(samples.astype(np.float32) / 32768.0)


def _write_pcm_wav(path: PathLike, samples: np.ndarray, sampling_rate: int) -> None:
    """Escribe samples int16 (1-D) a un WAV PCM mono 16-bit."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sampling_rate)
        wf.writeframes(samples.astype("<i2").tobytes())


# ---------------------------------------------------------------------------
# Post-proceso de timestamps: padding de 400 ms + fusión de solapes
# ---------------------------------------------------------------------------
def _pad_and_merge(
    raw_segments: Sequence[Tuple[float, float]],
    duration_s: float,
    pad_ms: int,
) -> List[Tuple[float, float]]:
    """Aplica `pad_ms` antes y después de cada tramo y fusiona los solapados.

    ``raw_segments``: tramos (inicio, fin) en segundos, sin orden garantizado
    (Silero ya los devuelve ordenados). El padding se recorta en los bordes
    del audio y los tramos consecutivos que se solapen se unen en uno solo.
    """
    if pad_ms < 0:
        raise ValueError(f"pad_ms no puede ser negativo: {pad_ms}")
    pad_s = pad_ms / 1000.0
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(raw_segments):
        start = max(0.0, start - pad_s)
        end = min(duration_s, end + pad_s)
        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def detect_speech_segments(
    input_path: PathLike,
    *,
    sampling_rate: int = DEFAULT_SAMPLE_RATE,
    threshold: float = DEFAULT_THRESHOLD,
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
    speech_pad_ms: int = DEFAULT_SPEECH_PAD_MS,
    max_speech_duration_s: float = float("inf"),
    neg_threshold: Optional[float] = None,
) -> List[SpeechSegment]:
    """Detecta los tramos de voz del audio normalizado y aplica el padding.

    Args:
        input_path: WAV PCM 16 kHz / mono / 16-bit (salida de la Fase 1).
        sampling_rate: Tasa del audio de entrada (debe ser 16000).
        threshold: Umbral de probabilidad de voz de Silero VAD (0.5 por defecto).
        min_speech_duration_ms: Descartar ráfagas de voz más cortas.
        min_silence_duration_ms: Silencio que cierra un tramo de voz.
        speech_pad_ms: Padding añadido **antes y después** de cada tramo
            (400 ms por defecto).
        max_speech_duration_s: Corta tramos de voz más largos que este límite.
        neg_threshold: Umbral de salida (None = threshold - 0.15 por defecto).

    Returns:
        Lista de :class:`SpeechSegment` ordenada, con timestamps en segundos
        y ya con el padding aplicado y fusionado.

    Raises:
        VadError: Si el archivo no existe o no cumple el formato normalizado.
    """
    wav = read_normalized_audio(input_path, sampling_rate)
    duration_s = wav.numel() / sampling_rate
    if wav.numel() < _VAD_WINDOW_SAMPLES:
        # Audio más corto que la ventana mínima de inferencia de Silero: sin voz.
        return []

    model = get_vad_model()
    try:
        from silero_vad import get_speech_timestamps
    except ModuleNotFoundError as exc:
        raise VadError(
            "La dependencia 'silero-vad' no está instalada. "
            "Ejecuta: pip install silero-vad"
        ) from exc
    raw_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=threshold,
        sampling_rate=sampling_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=0,  # el padding lo aplicamos nosotros (400 ms por lado)
        return_seconds=True,
        neg_threshold=neg_threshold,
    )
    raw_seconds = [(float(ts["start"]), float(ts["end"])) for ts in raw_timestamps]
    padded = _pad_and_merge(raw_seconds, duration_s, speech_pad_ms)
    return [SpeechSegment(start=start, end=end) for start, end in padded]


def segments_to_dicts(segments: Sequence[SpeechSegment]) -> List[Dict[str, float]]:
    """Serializa los segmentos a la estructura JSON `[{start, end, ...}, ...]`."""
    return [segment.to_dict() for segment in segments]


def cut_speech_chunk(
    input_path: PathLike,
    segment: SpeechSegment,
    output_path: PathLike,
    *,
    sampling_rate: int = DEFAULT_SAMPLE_RATE,
) -> str:
    """Corta y guarda un único chunk de voz (recorte exacto por muestra)."""
    samples = _read_pcm_samples(input_path, sampling_rate)
    start_sample = min(len(samples), int(round(segment.start * sampling_rate)))
    end_sample = min(len(samples), int(round(segment.end * sampling_rate)))
    end_sample = max(start_sample, end_sample)
    _write_pcm_wav(output_path, samples[start_sample:end_sample], sampling_rate)
    return os.path.abspath(output_path)


def export_speech_chunks(
    input_path: PathLike,
    segments: Sequence[SpeechSegment],
    out_dir: PathLike,
    *,
    prefix: str = "chunk_",
    sampling_rate: int = DEFAULT_SAMPLE_RATE,
) -> List[str]:
    """Exporta cada tramo de voz a un WAV independiente en ``out_dir``.

    Los nombres siguen el patrón ``{prefix}{idx:04d}_{start_ms:06d}_{end_ms:06d}.wav``
    para que quede reflejado el timestamp (ms) de cada fragmento.

    Returns:
        Lista de rutas absolutas de los chunks generados.
    """
    os.makedirs(str(out_dir), exist_ok=True)
    paths: List[str] = []
    for idx, segment in enumerate(segments, start=1):
        start_ms, end_ms = segment.to_ms()
        filename = f"{prefix}{idx:04d}_{start_ms:06d}_{end_ms:06d}.wav"
        out_path = os.path.join(str(out_dir), filename)
        paths.append(cut_speech_chunk(input_path, segment, out_path, sampling_rate=sampling_rate))
    return paths