# -*- coding: utf-8 -*-
"""E2E del pipeline: Fase 1 -> Fase 2 -> Fase 4 -> Fase 3.

Simula el flujo completo que ejecuta app.py sobre un archivo de prueba:
  * Fase 1: ``preprocess_audio_base`` (PCM 16 kHz / mono / 16-bit, -18 LUFS)
  * Fase 2: ``detect_speech_segments`` + ``export_speech_chunks`` (pad 400 ms)
  * Fase 4: ``detect_language_for_segment`` + ``detect_text_language`` (híbrido)
  * Fase 3: transcripción tramo a tramo USANDO una ASR local real como
    stand-in del LLM remoto, SIN idioma forzado (``language=None``: la ASR
    transcribe literalmente en el idioma original, evitando traducción).

Se valida la **consistencia global**: todos los timestamps quedan dentro del
audio original, los chunks exportados llevan los mismos ms en el nombre, y la
etiqueta de cada tramo refleja el idioma detectado (LID/híbrido).
"""

import os
import urllib.request
import wave

import numpy as np
import pytest

import audio_lid as lid
import audio_vad as vad
from audio_preprocessing import preprocess_audio_base

#: Muestra de referencia estable (habla inglesa real, 60 s).
SAMPLE_URL = "https://models.silero.ai/vad_models/en.wav"
SAMPLE_NAME = "en.wav"

MODEL_OK = None


def _models_ready() -> bool:
    """Comprueba la disponibilidad de los modelos VAD y LID (red/pip)."""
    global MODEL_OK
    if MODEL_OK is None:
        try:
            vad.get_vad_model()
            lid.get_lid_model()
            MODEL_OK = True
        except Exception:  # noqa: BLE001 - sin red o sin dependencias
            MODEL_OK = False
    return MODEL_OK


pytestmark = pytest.mark.skipif(
    not _models_ready(),
    reason="Modelos VAD/LID no disponibles (sin red o sin dependencias)",
)


def _download_or_skip(url: str, path) -> None:
    try:
        urllib.request.urlretrieve(url, str(path))
    except Exception as exc:  # noqa: BLE001 - sin red
        pytest.skip(f"Descarga de muestra no disponible: {exc}")


def _formatear_tiempo(segundos: float) -> str:
    """Mismo formato de acta que app.py: [MM:SS]."""
    total = int(segundos)
    return f"{total // 60:02d}:{total % 60:02d}"


def test_e2e_pipeline_completo(tmp_path):
    # ---------------------------------------------------------------- Fase 1
    src = tmp_path / SAMPLE_NAME
    _download_or_skip(SAMPLE_URL, src)
    normalized = tmp_path / "normalizado.wav"
    preprocess_audio_base(src, normalized)

    with wave.open(str(normalized), "rb") as wf:
        assert wf.getframerate() == 16_000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # PCM 16-bit

    from pydub import AudioSegment

    audio_total = AudioSegment.from_file(src)
    total_ms = len(audio_total)

    # ---------------------------------------------------------------- Fase 2
    segments = vad.detect_speech_segments(normalized)
    assert segments, "El VAD debe localizar al menos un tramo de voz"
    ranges = vad.speech_segments_to_ranges(segments)
    chunk_paths = vad.export_speech_chunks(
        normalized, segments, tmp_path / "chunks"
    )
    assert len(chunk_paths) == len(segments) == len(ranges)

    for (s_ms, e_ms), path, segment in zip(ranges, chunk_paths, segments):
        # Timestamps relativos al audio original (sincronía global)
        assert 0 <= s_ms < e_ms <= total_ms
        # Padding de 400 ms por lado ya aplicado
        assert segment.duration >= 0.8 - 1e-6
        # El chunk exportado refleja los mismos ms en su nombre
        name = os.path.basename(path)
        assert f"_{s_ms:06d}_" in name
        assert os.path.splitext(name)[0].endswith(f"_{e_ms:06d}")
        # Duración del chunk ≈ duración del tramo (grano de 1 muestra)
        with wave.open(str(path), "rb") as wf:
            samples = wf.getnframes()
        assert abs(samples / 16000.0 - segment.duration) < 0.001

    # ---------------------------------------------------------------- Fase 4
    lid_audio = lid.load_audio_array(normalized)
    annotated = lid.assign_languages(segments, lid_audio)
    forced_languages = [seg.language for seg in annotated]
    assert all(forced_languages), "Cada segmento debe tener idioma asignado"
    # En la muestra de referencia (inglés) el LID debe forzar 'en' con confianza
    for seg in annotated:
        _, detection = lid.detect_language_for_segment(lid_audio, seg)
        assert detection.confidence > 0.5

    # ------------------------------------------------------------- Fase 3
    # Stand-in del LLM: ASR local real. SIN language forzado (language=None)
    # para que transciba literalmente en el idioma original de cada tramo.
    model = lid.get_lid_model()
    acta_lines = []
    transcritos_con_texto = 0
    for (s_ms, e_ms), seg, language in zip(ranges, annotated, forced_languages):
        start_sample = int(round(seg.start * 16_000))
        end_sample = int(round(seg.end * 16_000))
        chunk = lid_audio[start_sample:end_sample]

        initial_prompt = (
            f"Transcripción bilingüe en castellano y {language.upper()} ({language}). "
            "Transcribir literalmente las palabras pronunciadas sin traducir ni normalizar."
        )
        result, _info = model.transcribe(  # language=None -> autodetección literal
            chunk, language=None, beam_size=1, vad_filter=False,
            initial_prompt=initial_prompt,
        )
        texto = " ".join(s.text for s in result).strip()
        if texto:
            transcritos_con_texto += 1

        # Etiqueta con los MISMOS ms que app.py (formatear_tiempo(start_ms))
        acta_lines.append(
            f"[{_formatear_tiempo(s_ms / 1000.0)}] [{language.upper()}] {texto}"
        )

    # Coherencia idiomática: idioma del tramo == idioma forzado en la ASR
    assert all(language == "en" for language in forced_languages)
    assert transcritos_con_texto >= max(1, len(acta_lines) // 2), (
        "Al menos la mitad de los tramos deben transcribir texto (muestra con habla)"
    )

    # Acta: el primer tramo arranca en el segundo 0 y los idiomas quedan etiquetados
    acta = "\n".join(acta_lines)
    assert "[00:00]" in acta
    assert "[EN]" in acta
    # Cada línea del acta tiene su timestamp sincronizado con el audio original
    for (s_ms, _e_ms), line in zip(ranges, acta_lines):
        assert f"{_formatear_tiempo(s_ms / 1000.0)}" in line