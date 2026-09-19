# -*- coding: utf-8 -*-
"""Fase 4: Detección de idioma por segmento (LID) agnóstica y multilingüe.

Estrategia (context7 / faster-whisper y Whisper): usar el **selector de
idioma nativo de Whisper** sobre el propio chunk de audio
(``WhisperModel.detect_language(audio)``), que analiza los primeros segundos
del segmento y devuelve el código ISO del idioma (es, it, en, fr, de, pt,
ja...) junto con su probabilidad. Es la vía más ligera en CPU/ARM porque solo
ejecuta el encoder del modelo (sin decodificar texto) y corre sobre
CTranslate2 (``compute_type="int8"``), sin depender de torch.

Flujo:
  1. ``detect_language(audio)`` -> :class:`LanguageDetection` con el idioma
     ISO y el score de confianza (0..1).
  2. ``resolve_language(...)`` -> aplica el **umbral de confianza**: si el
     score del idioma detectado es inferior al umbral (o la detección falla),
     hace fallback al idioma por defecto configurado (``ASR_DEFAULT_LANGUAGE``,
     por defecto ``es``) para un comportamiento seguro.
  3. La app fuerza ``language=<idioma>`` al motor ASR por segmento evitando
     alucinaciones o traducciones no deseadas.

Configuración (variables de entorno):
  * ``ASR_LID_MODEL``             : tamaño del modelo Whisper para LID
                                   (por defecto ``tiny``; opcionales base/small).
  * ``ASR_LID_MODEL_DIR``         : carpeta del modelo precargado (opcional).
  * ``ASR_LID_CONFIDENCE_THRESHOLD``: umbral de confianza (por defecto 0.5).
  * ``ASR_DEFAULT_LANGUAGE``      : idioma por defecto/fallback (por defecto ``es``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from audio_vad import SpeechSegment, VadError, _read_pcm_samples

logger = logging.getLogger(__name__)

#: Idioma por defecto si el LID es poco confiable o falla (fallback seguro).
DEFAULT_LANGUAGE: str = os.getenv("ASR_DEFAULT_LANGUAGE", "es").strip().lower()
#: Umbral de confianza: por debajo de él se usa el idioma por defecto.
CONFIDENCE_THRESHOLD: float = float(os.getenv("ASR_LID_CONFIDENCE_THRESHOLD", "0.5"))
#: Tamaño del modelo Whisper usado únicamente para detectar el idioma.
LID_MODEL_SIZE: str = os.getenv("ASR_LID_MODEL", "tiny").strip()
#: Directorio opcional con el modelo ya descargado.
LID_MODEL_DIR: Optional[str] = os.getenv("ASR_LID_MODEL_DIR") or None

#: Ventana de análisis de Whisper (el LID usa los primeros segundos del chunk).
LID_FIRST_SECONDS: float = 30.0

ArrayLike = Union[str, os.PathLike, np.ndarray]


@dataclass(frozen=True)
class LanguageDetection:
    """Resultado de la detección de idioma de un segmento.

    Attributes:
        language: Código ISO-639-1 del idioma detectado (p. ej. ``es``, ``it``).
        confidence: Score de confianza en el rango [0, 1].
        probabilities: Probabilidades por idioma (top-N) ordenadas por score.
    """

    language: str = DEFAULT_LANGUAGE
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)

    def is_confident(self, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
        """True si el idioma detectado supera el umbral de confianza."""
        return bool(self.language) and self.confidence >= threshold


def normalize_iso(language: Optional[str]) -> str:
    """Normaliza el código devuelto por Whisper a ISO-639-1 en minúsculas."""
    if not language:
        return DEFAULT_LANGUAGE
    code = language.strip().lower()
    return code if code else DEFAULT_LANGUAGE


# ---------------------------------------------------------------------------
# Modelo (carga perezosa + caché)
# ---------------------------------------------------------------------------
_lid_model = None


def get_lid_model():
    """Carga (una sola vez) el modelo Whisper ligero para LID (CPU/int8)."""
    global _lid_model
    if _lid_model is None:
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            raise LIDError(
                "La dependencia 'faster-whisper' no está instalada "
                "(pip install faster-whisper)."
            ) from exc
        _lid_model = WhisperModel(
            LID_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
            download_root=LID_MODEL_DIR,
        )
    return _lid_model


class LIDError(RuntimeError):
    """Error controlado del módulo de detección de idioma."""


# ---------------------------------------------------------------------------
# Lectura de audio (reutiliza la validación estricta de la Fase 2)
# ---------------------------------------------------------------------------
def load_audio_array(input_path, sampling_rate: int = 16_000) -> np.ndarray:
    """Lee un WAV PCM 16 kHz/mono/16-bit (salida Fase 1) como float32 [-1, 1]."""
    samples = _read_pcm_samples(input_path, sampling_rate)
    return (samples.astype(np.float32) / 32768.0)


def _to_audio_array(input_audio: ArrayLike, sampling_rate: int = 16_000) -> np.ndarray:
    """Normaliza la entrada (ruta o array float32 1-D) a float32 mono 16 kHz.

    Para rutas se usa ``decode_audio`` de faster-whisper (soporta mp3/flac/wav
    y re-muestrea a 16 kHz); si no está disponible, se cae al lector WAV
    estricto del pipeline (16 kHz/mono/16-bit, salida de la Fase 1).
    """
    if isinstance(input_audio, (str, os.PathLike)):
        try:
            from faster_whisper.audio import decode_audio

            return np.asarray(
                decode_audio(str(input_audio), sampling_rate=sampling_rate),
                dtype=np.float32,
            )
        except Exception:  # noqa: BLE001 - formato no soportado por PyAV -> WAV estricto
            return load_audio_array(input_audio, sampling_rate)
    audio = np.asarray(input_audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    if not audio.size:
        raise LIDError("Array de audio vacío para la detección de idioma.")
    return audio


# ---------------------------------------------------------------------------
# Detección
# ---------------------------------------------------------------------------
def detect_language(
    input_audio: ArrayLike,
    *,
    model=None,
    top_n: int = 5,
    sampling_rate: int = 16_000,
) -> LanguageDetection:
    """Detecta el idioma de un chunk de audio con el selector nativo de Whisper.

    Args:
        input_audio: Ruta a un WAV 16 kHz/mono o array numpy float32 mono.
        model: Modelo ya cargado (si no, se carga/cachea uno Tiny).
        top_n: Nº de idiomas a conservar en ``probabilities``.
        sampling_rate: Tasa de muestreo del audio de entrada.

    Returns:
        :class:`LanguageDetection` con el ISO del idioma y la confianza.
        Si la detección falla, devuelve el idioma por defecto con confianza 0
        (el fallback lo resuelve :func:`resolve_language`).
    """
    try:
        audio = _to_audio_array(input_audio, sampling_rate)
        model = model or get_lid_model()
        language, probability, all_probs = model.detect_language(audio, vad_filter=False)
    except Exception as exc:  # noqa: BLE001 - LID nunca debe romper la transcripción
        logger.warning("LID falló (%s); se usará el idioma por defecto.", exc)
        return LanguageDetection(language=DEFAULT_LANGUAGE, confidence=0.0)

    probabilities = {
        normalize_iso(code): float(prob)
        for code, prob in sorted(all_probs or [], key=lambda item: item[1], reverse=True)[:top_n]
    }
    return LanguageDetection(
        language=normalize_iso(language),
        confidence=float(probability or 0.0),
        probabilities=probabilities,
    )


def detect_language_for_segment(
    audio_array: np.ndarray,
    segment: SpeechSegment,
    *,
    model=None,
    sampling_rate: int = 16_000,
    threshold: float = CONFIDENCE_THRESHOLD,
    default: str = DEFAULT_LANGUAGE,
) -> Tuple[str, LanguageDetection]:
    """Detecta el idioma de un tramo concreto y devuelve (idioma, detección).

    Corta el array según los límites del :class:`SpeechSegment` y, si la
    confianza no supera ``threshold``, devuelve el idioma ``default``.
    """
    start_sample = min(len(audio_array), int(round(segment.start * sampling_rate)))
    end_sample = min(len(audio_array), int(round(segment.end * sampling_rate)))
    end_sample = max(start_sample, end_sample)
    chunk = audio_array[start_sample:end_sample]
    detection = detect_language(chunk, model=model, sampling_rate=sampling_rate)
    language, _ = resolve_language(detection, threshold=threshold, default=default)
    return language, detection


# ---------------------------------------------------------------------------
# Umbral / fallback
# ---------------------------------------------------------------------------
def resolve_language(
    detection: LanguageDetection,
    *,
    threshold: float = CONFIDENCE_THRESHOLD,
    default: str = DEFAULT_LANGUAGE,
) -> Tuple[str, bool]:
    """Aplica el umbral de confianza con fallback al idioma por defecto.

    Returns:
        ``(idioma, confianza_ok)``: si la confianza del idioma detectado es
        inferior a ``threshold`` o la detección está vacía, se devuelve
        ``(default, False)`` (comportamiento seguro: la ASR usa el idioma
        global configurado o su autodetección).
    """
    if detection.is_confident(threshold) and detection.language:
        return normalize_iso(detection.language), True
    return normalize_iso(default), False


# ---------------------------------------------------------------------------
# Anotación de segmentos
# ---------------------------------------------------------------------------
def assign_languages(
    segments: Sequence[SpeechSegment],
    audio_array: Optional[np.ndarray] = None,
    *,
    model=None,
    sampling_rate: int = 16_000,
    threshold: float = CONFIDENCE_THRESHOLD,
    default: str = DEFAULT_LANGUAGE,
) -> List[SpeechSegment]:
    """Devuelve una copia de los segmentos con ``language`` asignado por LID.

    Si ``audio_array`` no se proporciona, los segmentos se anotan con el
    idioma por defecto (utilitario para flujos sin audio normalizado).
    """
    annotated: List[SpeechSegment] = []
    for segment in segments:
        if audio_array is None:
            annotated.append(segment.with_language(default))
            continue
        language, _ = detect_language_for_segment(
            audio_array, segment, model=model,
            sampling_rate=sampling_rate, threshold=threshold, default=default,
        )
        annotated.append(segment.with_language(language))
    return annotated