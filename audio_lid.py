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
  3. **La ASR transcribe SIN idioma forzado** (la transcripción es literal en
     el idioma original hablado, evitando traducciones inducidas por Whisper).
     El idioma final del fragmento lo decide la **decisión híbrida audio+texto**
     con :func:`detect_text_language` (Español vs Lengua B de la sesión).

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
from dataclasses import dataclass, field, replace
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

#: Duración mínima (s) para ejecutar el LID de Whisper: los fragmentos MÁS
#: cortos no tienen contexto acústico suficiente y **no se analizan**; heredan
#: el idioma del segmento anterior evitando caídas al idioma por defecto.
MIN_LID_DURATION_S: float = 1.5
#: Segundos de audio circundante (a cada lado) para ampliar el análisis de
#: un tramo de baja confianza (que ya tiene duración suficiente).
CONTEXT_EXTRA_SECONDS: float = 1.0

# --- Matriz de transición temporal (inercia de idioma según el silencio) -------
#: Ventana de inercia conversacional ``T_inertia`` (s): si el intervalo de
#: silencio entre fragmentos contiguos es menor (por defecto 6.0 s: pausas de
#: pensamiento/respiración en turnos naturales de 3-6 s), se considera el MISMO
#: turno y se premia/hereda el idioma precedente. Configurable con la env
#: ``ASR_INERTIA_WINDOW_S`` (legado: ``ASR_LID_T_INERTIA``).
T_INERTIA_S: float = float(os.getenv("ASR_INERTIA_WINDOW_S",
                                     os.getenv("ASR_LID_T_INERTIA", "6.0")))
#: Bonificación suave (inertia bias) sumada a la probabilidad del idioma del
#: segmento anterior antes de seleccionar el máximo dentro de allowed_languages.
INERTIA_BIAS: float = float(os.getenv("ASR_LID_INERTIA_BIAS", "0.15"))
#: Histéresis de turno (delta sobre el umbral): para CAMBIAR de idioma en medio
#: de un turno activo (Δt < T_inertia) se exige evidencia abrumadora
#: (``score >= threshold + delta``). Configurable con ``ASR_LID_HYSTERESIS_DELTA``.
HYSTERESIS_DELTA: float = float(os.getenv("ASR_LID_HYSTERESIS_DELTA", "0.15"))
#: Confianza de SALIDA (doble confirmación acústica): dentro de un turno activo
#: cambiar de idioma respecto al precedente exige
#: ``score >= max(threshold + HYSTERESIS_DELTA, EXIT_CONFIDENCE)`` (por defecto
#: 0.95). Evita salir de la Lengua B con pronunciación castellanizada.
#: Configurable con ``ASR_LID_EXIT_CONFIDENCE``.
EXIT_CONFIDENCE: float = float(os.getenv("ASR_LID_EXIT_CONFIDENCE", "0.95"))

#: Variable de entorno con la lista de idiomas candidatos (ISO-639-1, comas).
_ENV_ALLOWED_LANGUAGES: str = "ASR_ALLOWED_LANGUAGES"

ArrayLike = Union[str, os.PathLike, np.ndarray]


def parse_allowed_languages(value: Optional[str] = None) -> Optional[List[str]]:
    """Parsea la lista de idiomas permitidos (códigos ISO, separados por coma).

    Si no se pasa valor ni está definida ``ASR_ALLOWED_LANGUAGES``, devuelve
    ``None`` (sin restricción: el LID evalúa todos los idiomas soportados).
    """
    if value is None:
        value = os.getenv(_ENV_ALLOWED_LANGUAGES, "")
    codes = [
        normalize_iso(item)
        for item in str(value).split(",")
        if item and item.strip()
    ]
    return codes or None


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
    context_window_s: float = CONTEXT_EXTRA_SECONDS,
    previous_language: Optional[str] = None,
    previous_end: Optional[float] = None,
    allowed_languages: Optional[Sequence[str]] = None,
    t_inertia: Optional[float] = None,
    hysteresis_delta: Optional[float] = None,
) -> Tuple[str, LanguageDetection]:
    """Detecta el idioma de un tramo con suavizado contextual e inercia temporal.

    Estrategia (fragmentos cortos o de baja confianza LID):
      * Si el tramo dura menos de ``MIN_LID_DURATION_S`` (1.5 s), **NO se
        ejecuta** Whisper LID (no hay contexto acústico suficiente): se hereda
        el idioma del segmento anterior (``previous_language``) o, si no hay,
        ``default``. Evita caídas al idioma por defecto en micro-chunks VAD.
      * Si el tramo tiene duración suficiente pero el score (restringido a
        ``allowed_languages`` si está definido) no supera ``threshold``, se
        **amplía la ventana de análisis** con ``context_window_s`` segundos
        (±1 s) del audio circundante.
      * **Inercia temporal:** si ``previous_end`` es conocido, se calcula el
        intervalo de silencio ``Δt = segment.start − previous_end``. Con
        ``Δt < t_inertia`` (por defecto ``T_INERTIA_S`` = 6.0 s) se suma
        ``INERTIA_BIAS`` a la probabilidad del idioma anterior
        (``previous_language``) antes de elegir el máximo dentro de
        ``allowed_languages``; con ``Δt ≥ t_inertia`` la inercia se anula y la
        evaluación es neutra.
      * **Histéresis de turno:** dentro de la ventana (``Δt < t_inertia``), si el
        idioma precedente está activo y el candidato propone CAMBIAR de idioma
        (p. ej. Lengua B → español a mitad de turno), se exige evidencia
        **abrumadora**: ``score >= threshold + hysteresis_delta``
        (por defecto 0.65 con threshold 0.5). Si no se alcanza, se MANTIENE la
        inercia del idioma activo en lugar de caer al idioma por defecto.
      * Si la confianza sigue por debajo del umbral, se **hereda el idioma del
        segmento anterior** si y solo si ``Δt < t_inertia`` (en silencios
        largos se cae al idioma por defecto, sin inercia).

    Cuando ``allowed_languages`` no es None, se solicita al detector el mapa
    de probabilidades completo (top-N mayor) para poder filtrar por las
    claves del subconjunto permitido y elegir el mejor candidato de él.

    Returns:
        ``(idioma, detección)``: el idioma resuelto (ISO) y la última
        :class:`LanguageDetection` (para registrar su confianza). En
        micro-chunks sin análisis, la detección devuelve confianza 0.0.
    """
    start_sample = min(len(audio_array), int(round(segment.start * sampling_rate)))
    end_sample = min(len(audio_array), int(round(segment.end * sampling_rate)))
    end_sample = max(start_sample, end_sample)
    delta_t = segment.start - previous_end if previous_end is not None else None
    window = t_inertia if t_inertia is not None else T_INERTIA_S
    hdelta = hysteresis_delta if hysteresis_delta is not None else HYSTERESIS_DELTA

    # Micro-chunk: sin contexto acústico suficiente -> heredar, sin ejecutar LID.
    if segment.duration < MIN_LID_DURATION_S:
        inherited = normalize_iso(previous_language) if previous_language else normalize_iso(default)
        return inherited, LanguageDetection(language=inherited, confidence=0.0)

    top_n = 99 if allowed_languages else 5
    detection = detect_language(
        audio_array[start_sample:end_sample], model=model,
        sampling_rate=sampling_rate, top_n=top_n,
    )
    detection = _bias_by_time_inertia(detection, previous_language, delta_t,
                                      t_inertia=window)
    candidate, score = _restricted_best(detection, allowed_languages)
    if score < threshold and context_window_s > 0:
        context_samples = int(round(context_window_s * sampling_rate))
        context_start = max(0, start_sample - context_samples)
        context_end = min(len(audio_array), end_sample + context_samples)
        detection = detect_language(
            audio_array[context_start:context_end],
            model=model, sampling_rate=sampling_rate, top_n=top_n,
        )
        detection = _bias_by_time_inertia(detection, previous_language, delta_t,
                                          t_inertia=window)
        candidate, score = _restricted_best(detection, allowed_languages)

    # --- Decisión con HISTÉRESIS DE TURNO (agnóstica a idiomas) -------------
    en_ventana = _en_ventana(previous_language, delta_t, window)
    prev = normalize_iso(previous_language) if previous_language else None
    if candidate is not None and score >= threshold:
        # Doble confirmación acústica de salida del idioma activo en el turno.
        umbral_cambio = max(threshold + hdelta, EXIT_CONFIDENCE)
        # Dentro de un turno activo, cambiar de idioma exige evidencia abrumadora.
        if en_ventana and allowed_languages and prev is not None and normalize_iso(candidate) != prev:
            if score < umbral_cambio:
                return normalize_iso(prev), detection
            return normalize_iso(candidate), detection
        return normalize_iso(candidate), detection
    if en_ventana and prev:
        # Herencia: se mantiene la inercia del idioma activo del turno.
        return normalize_iso(prev), detection
    return normalize_iso(default), detection


# ---------------------------------------------------------------------------
# Clasificación de texto agnóstica (solo ES + Lengua B, sin léxico harcodeado)
# ---------------------------------------------------------------------------
#: Backend NLP del clasificador de texto: ``lingua-language-detector``
#: (recomendado) con fallback a ``langdetect`` (100 % Python, útil en ARM sin
#: wheel de lingua). Se configura con la env ``ASR_TEXT_LID_BACKEND``.
_ENV_TEXT_LID_BACKEND: str = "ASR_TEXT_LID_BACKEND"
_text_lid_detectors: Dict[tuple, object] = {}


def _build_lingua_detector(languages: Sequence[str]):
    """Detector lingua restringido a los idiomas dados (códigos ISO dinámicos)."""
    from lingua import IsoCode639_1, LanguageDetectorBuilder

    enums = [getattr(IsoCode639_1, normalize_iso(code).upper()) for code in languages if code]
    detector = LanguageDetectorBuilder.from_iso_codes_639_1(*enums).build()

    def detect_text(texto: str):
        language = detector.detect_language_of(texto)
        if language is None:
            return None
        confidence = float(detector.compute_language_confidence(texto, language))
        return normalize_iso(language.iso_code_639_1.name), confidence

    return detect_text


def _build_langdetect_detector(languages: Sequence[str]):
    """Detector langdetect restringido a los idiomas dados (fallback puro Python)."""
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0  # determinista
    allowed = {normalize_iso(code) for code in languages if code}

    def detect_text(texto: str):
        try:
            hits = detect_langs(texto)
        except Exception:  # noqa: BLE001 - texto demasiado corto/ilegible
            return None
        for item in hits:
            code = normalize_iso(item.lang)
            if code in allowed:
                return code, float(item.prob)
        return None

    return detect_text


def _text_detector_for(languages: Sequence[str]):
    """Devuelve la función texto->(código_iso, confianza) para los idiomas dados."""
    def _build():
        preference = os.getenv(_ENV_TEXT_LID_BACKEND, "auto").strip().lower()
        order = ("lingua", "langdetect") if preference == "auto" else ((preference,) if preference in ("lingua", "langdetect") else ())
        for backend in order:
            try:
                if backend == "lingua":
                    return _build_lingua_detector(languages)
                if backend == "langdetect":
                    return _build_langdetect_detector(languages)
            except Exception:  # noqa: BLE001 - backend no disponible: probar el siguiente
                continue
        return None

    key = tuple(sorted(normalize_iso(code) for code in languages if code))
    if key not in _text_lid_detectors:
        _text_lid_detectors[key] = _build()
    return _text_lid_detectors[key]


def detect_text_language(
    texto: Optional[str],
    iso_a: str = DEFAULT_LANGUAGE,
    iso_b: str = "",
    *,
    min_confidence: float = CONFIDENCE_THRESHOLD,
) -> Optional[str]:
    """Clasifica el idioma del texto ÚNICAMENTE entre ``iso_a`` e ``iso_b``.

    Arquitectura **100 % agnóstica** (sin palabras harcodeadas en ningún
    idioma): se usa un backend NLP (lingua-language-detector o langdetect)
    restringido a los dos idiomas de la sesión, p. ej. ``['es', 'it']``.

    Devuelve el código ISO detectado si la confianza supera ``min_confidence``;
    ``None`` si no hay texto, no hay backend NLP disponible o la confianza es
    baja (el llamador mantiene entonces el resultado del LID de audio).
    """
    codes = sorted({normalize_iso(code) for code in (iso_a, iso_b) if code})
    if not texto or not texto.strip() or len(codes) < 2:
        return None
    detector = _text_detector_for(codes)
    if detector is None:
        return None
    try:
        result = detector(texto[:2000])
    except Exception:  # noqa: BLE001 - el clasificador de texto nunca debe romper
        return None
    if not result:
        return None
    language, confidence = result
    if language in codes and confidence >= min_confidence:
        return language
    return None


# ---------------------------------------------------------------------------
# Umbral / fallback / restricción de idiomas
# ---------------------------------------------------------------------------
def _en_ventana(previous_language: Optional[str], delta_t: Optional[float],
                t_inertia: float) -> bool:
    """True si hay idioma previo y el silencio Δt cae dentro de la ventana."""
    return (previous_language is not None and delta_t is not None
            and delta_t < t_inertia)


def _bias_by_time_inertia(
    detection: LanguageDetection,
    previous_language: Optional[str],
    delta_t: Optional[float],
    *,
    t_inertia: Optional[float] = None,
) -> LanguageDetection:
    """Aplica el sesgo de inercia temporal a las probabilidades del detector.

    Si ``delta_t`` (intervalo de silencio desde el segmento anterior) es
    conocido y menor que ``t_inertia`` (por defecto ``T_INERTIA_S``), suma
    ``INERTIA_BIAS`` a la probabilidad del idioma del segmento anterior antes
    de elegir el máximo dentro de ``allowed_languages``. Si
    ``delta_t >= t_inertia`` (silencio largo), devuelve las probabilidades sin
    tocar (inercia anulada).
    """
    window = t_inertia if t_inertia is not None else T_INERTIA_S
    if previous_language is None or delta_t is None or delta_t >= window:
        return detection
    if not detection.probabilities:
        return detection
    previous = normalize_iso(previous_language)
    biased = dict(detection.probabilities)
    biased[previous] = float(biased.get(previous, 0.0)) + INERTIA_BIAS
    return replace(detection, probabilities=biased)


def _restricted_best(
    detection: LanguageDetection,
    allowed_languages: Optional[Sequence[str]],
) -> Tuple[Optional[str], float]:
    """Mejor idioma (y score) limitado a ``allowed_languages``.

    Con ``allowed_languages`` None/vacío devuelve el idioma global del
    detector con su confianza (comportamiento por defecto). El resto de
    idiomas (raros o secundarios) se descartan para evitar falsos positivos.
    """
    if not allowed_languages:
        return detection.language or None, detection.confidence
    allowed = {normalize_iso(code) for code in allowed_languages if code}
    restricted = {
        normalize_iso(code): float(score)
        for code, score in detection.probabilities.items()
        if normalize_iso(code) in allowed
    }
    if not restricted:
        return None, 0.0
    best = max(restricted, key=restricted.get)
    return best, restricted[best]


def resolve_language(
    detection: LanguageDetection,
    *,
    threshold: float = CONFIDENCE_THRESHOLD,
    default: str = DEFAULT_LANGUAGE,
    allowed_languages: Optional[Sequence[str]] = None,
) -> Tuple[str, bool]:
    """Aplica el umbral de confianza (y la restricción de idiomas) con fallback.

    Args:
        detection: Resultado del detector (con ``probabilities`` completo).
        threshold: Umbral de confianza para aceptar la detección.
        default: Idioma por defecto cuando el LID no es concluyente.
        allowed_languages: Códigos ISO candidatos. Si está definido, se
            selecciona el idioma con mayor score DENTRO de ese subconjunto
            (ignorando idiomas raros/secundarios). None/vacío = sin restricción.

    Returns:
        ``(idioma, confianza_ok)``: si la confianza del idioma (restringido o
        no) es inferior a ``threshold`` o no hay candidatos, se devuelve
        ``(default, False)`` (comportamiento seguro: la ASR usa el idioma
        global configurado o su autodetección).
    """
    language, score = _restricted_best(detection, allowed_languages)
    if language and score >= threshold:
        return normalize_iso(language), True
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
    allowed_languages: Optional[Sequence[str]] = None,
) -> List[SpeechSegment]:
    """Devuelve una copia de los segmentos con ``language`` asignado por LID.

    Si ``audio_array`` no se proporciona, los segmentos se anotan con el
    idioma por defecto (utilitario para flujos sin audio normalizado).

    ``allowed_languages`` restringe los candidatos del LID (None/vacío =
    evaluación completa sobre todos los idiomas soportados).
    """
    annotated: List[SpeechSegment] = []
    previous_language: Optional[str] = None
    previous_segment: Optional[SpeechSegment] = None
    for segment in segments:
        if audio_array is None:
            annotated.append(segment.with_language(default))
            previous_language = previous_language or default
            previous_segment = previous_segment or segment
            continue
        language, detection = detect_language_for_segment(
            audio_array, segment, model=model,
            sampling_rate=sampling_rate, threshold=threshold, default=default,
            previous_language=previous_language,
            previous_end=previous_segment.end if previous_segment is not None else None,
            allowed_languages=allowed_languages,
        )
        annotated.append(segment.with_language(language))
        # El idioma "anterior" pasa a ser el resuelto: así la inercia temporal
        # (Δt < T_inertia) y la herencia operan sobre el último tramo tratado.
        previous_language = language
        previous_segment = segment
    return annotated