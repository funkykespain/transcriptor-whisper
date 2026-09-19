# -*- coding: utf-8 -*-
"""Tests del módulo de detección de idioma por segmento (`audio_lid.py`).

Cubren: (1) lógica pura de umbral/fallback y normalización ISO, (2) el corte
por segmento y la anotación de ``SpeechSegment.language`` (con el detector
simulado para es/it), (3) la validación del formato de entrada y (4) la
detección real con Whisper sobre muestras multilingües (OJO: requiere red la
primera vez y se omiten si el modelo no está disponible).
"""

import os
import urllib.request
import wave

import numpy as np
import pytest

import audio_lid as lid
from audio_lid import (
    DEFAULT_LANGUAGE,
    LanguageDetection,
    resolve_language,
)
from audio_vad import SpeechSegment

MODEL_READY = None


def _model_is_ready() -> bool:
    global MODEL_READY
    if MODEL_READY is None:
        try:
            lid.get_lid_model()
            MODEL_READY = True
        except Exception:  # noqa: BLE001 - sin red/deps: se omiten esos tests
            MODEL_READY = False
    return MODEL_READY


requires_model = pytest.mark.skipif(
    not _model_is_ready(), reason="Modelo Whisper LID no disponible (sin red o sin dependencias)"
)


def _write_wav(path, samples, sample_rate=16000, n_channels=1, sampwidth=2):
    samples = np.asarray(samples)
    pcm = np.clip(samples, -1.0, 1.0) * (2 ** (8 * sampwidth - 1) - 1)
    pcm = pcm.astype("<i2" if sampwidth == 2 else "<i1").tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _tone(duration_s, sample_rate=16000, freq=440.0, amp=0.3):
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    return amp * np.sin(2 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# 1) Lógica pura: umbral de confianza, fallback e ISO
# ---------------------------------------------------------------------------
def test_resolve_language_usa_idioma_detectado_si_confia():
    lang, ok = resolve_language(LanguageDetection(language="it", confidence=0.9))
    assert lang == "it"
    assert ok is True


def test_resolve_language_fallback_si_confianza_baja():
    lang, ok = resolve_language(
        LanguageDetection(language="en", confidence=0.3),
        threshold=0.5,
        default="es",
    )
    assert lang == "es"          # fallback al idioma global configurado
    assert ok is False


def test_resolve_language_fallback_si_deteccion_vacia():
    lang, ok = resolve_language(
        LanguageDetection(language="", confidence=0.0), default="es"
    )
    assert lang == "es"
    assert ok is False


def test_is_confident_en_el_umbral():
    assert LanguageDetection(language="it", confidence=0.5).is_confident(0.5) is True
    assert LanguageDetection(language="it", confidence=0.499).is_confident(0.5) is False


def test_normalize_iso():
    assert lid.normalize_iso("EN") == "en"
    assert lid.normalize_iso(None) == DEFAULT_LANGUAGE


def test_detect_language_probar_es_vs_it_con_detector_simulado(monkeypatch):
    """El discriminador es/it (lógica del umbral) con detección simulada."""
    calls = {"langs": ["it", "es", "it", "es"], "confs": [0.95, 0.98, 0.2, 0.1]}
    idx = {"n": 0}

    def fake_detect(audio, **kwargs):
        n = idx["n"]
        idx["n"] += 1
        return LanguageDetection(
            language=calls["langs"][n], confidence=calls["confs"][n]
        )

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 4, dtype=np.float32)
    segments = [SpeechSegment(0.5, 1.5).with_language("es"), SpeechSegment(2.5, 3.5).with_language("es")]
    annotated = lid.assign_languages(segments, audio_array)
    assert [s.language for s in annotated] == ["it", "es"]  # 4º con conf. baja -> es


# ---------------------------------------------------------------------------
# 2) Corte por segmento y formato
# ---------------------------------------------------------------------------
def test_detect_language_for_segment_corta_y_aplica_umbral(monkeypatch):
    captured = {}

    def fake_detect(audio, **kwargs):
        captured["len"] = len(audio)
        return LanguageDetection(language="it", confidence=0.9)

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 3, dtype=np.float32)
    lang, det = lid.detect_language_for_segment(
        audio_array, SpeechSegment(0.5, 1.5), default="es"
    )
    assert lang == "it"
    assert captured["len"] == 16000  # 1.0 s cortado del array completo


def test_detect_language_formato_no_normalizado_lanza(tmp_path):
    bad = tmp_path / "bad_44k.wav"
    _write_wav(bad, _tone(1.0), sample_rate=44100)
    with pytest.raises(lid.VadError):
        lid.load_audio_array(bad)


def test_detect_language_archivo_inexistente_no_rompe():
    det = lid.detect_language("/no/existe.wav")
    assert det.language == DEFAULT_LANGUAGE
    assert det.confidence == 0.0


def test_detect_language_array_vacio_devuelve_fallback():
    det = lid.detect_language(np.array([], dtype=np.float32))
    assert det.language == DEFAULT_LANGUAGE


# ---------------------------------------------------------------------------
# 3) Integración con Whisper (muestras reales en varios idiomas)
# ---------------------------------------------------------------------------
def _urlretrieve_or_skip(url, path):
    try:
        urllib.request.urlretrieve(url, str(path))
    except Exception as exc:  # noqa: BLE001 - sin red: omitimos
        pytest.skip(f"Sin red o URL no disponible ({exc})")


@requires_model
def test_detect_language_audio_real_en(tmp_path):
    src = tmp_path / "en.wav"
    _urlretrieve_or_skip("https://models.silero.ai/vad_models/en.wav", src)
    det = lid.detect_language(str(src))
    assert det.language == "en"
    assert det.confidence > 0.5


@requires_model
def test_detect_language_audio_real_multilingue(tmp_path):
    """Muestra oficial multilingüe de faster-whisper: devuelve ISO válido + score."""
    src = tmp_path / "multilingual.mp3"
    _urlretrieve_or_skip(
        "https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/tests/data/multilingual.mp3",
        src,
    )
    det = lid.detect_language(str(src))
    assert lid.normalize_iso(det.language) == det.language
    assert 0.0 < det.confidence <= 1.0
    assert det.language in det.probabilities


@requires_model
def test_detect_language_tono_sintetico_cae_a_fallback():
    """Un tono (no habla) no supera el umbral -> fallback al idioma por defecto."""
    audio = _tone(2.0).astype(np.float32)
    det = lid.detect_language(audio)
    lang, ok = resolve_language(det, default="es")
    assert lang == "es"
    assert ok is False or det.confidence >= 0.5


@requires_model
def test_assign_languages_pipeline_completo_en_audio_real(tmp_path):
    """Fase 1 -> VAD -> LID: cada segmento queda anotado con un ISO válido."""
    from audio_preprocessing import preprocess_audio_base
    from audio_vad import detect_speech_segments

    src = tmp_path / "en.wav"
    _urlretrieve_or_skip("https://models.silero.ai/vad_models/en.wav", src)
    normalized = tmp_path / "norm.wav"
    preprocess_audio_base(src, normalized)

    segments = detect_speech_segments(normalized)
    assert segments
    audio_array = lid.load_audio_array(normalized)
    annotated = lid.assign_languages(segments, audio_array)

    assert len(annotated) == len(segments)
    for seg in annotated:
        assert seg.language  # ISO no vacío
        assert seg.language == lid.normalize_iso(seg.language)