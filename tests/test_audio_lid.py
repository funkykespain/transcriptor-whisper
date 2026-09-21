# -*- coding: utf-8 -*-
"""Tests del módulo de detección de idioma por segmento (`audio_lid.py`).

Cubren: (1) lógica pura de umbral/fallback y normalización ISO, (2) el corte
por segmento y la anotación de ``SpeechSegment.language`` (con el detector
simulado para es/it), (3) la validación del formato de entrada y (4) la
detección real con Whisper sobre muestras multilingües (OJO: requiere red la
primera vez y se omiten si el modelo no está disponible).
"""

import importlib.util
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
    calls = {"langs": ["it", "es"], "confs": [0.95, 0.98]}
    idx = {"n": 0}

    def fake_detect(audio, **kwargs):
        n = idx["n"]
        idx["n"] += 1
        return LanguageDetection(
            language=calls["langs"][n], confidence=calls["confs"][n]
        )

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 6, dtype=np.float32)
    # Tramos >= 1.5 s para no disparar la expansión de contexto del LID.
    segments = [SpeechSegment(0.5, 2.5).with_language("es"), SpeechSegment(3.0, 5.0).with_language("es")]
    annotated = lid.assign_languages(segments, audio_array)
    assert [s.language for s in annotated] == ["it", "es"]


# ---------------------------------------------------------------------------
# 2) Corte por segmento, formato y duración mínima de LID
# ---------------------------------------------------------------------------
def test_detect_language_for_segment_corta_y_aplica_umbral(monkeypatch):
    captured = {}

    def fake_detect(audio, **kwargs):
        captured["len"] = len(audio)
        return LanguageDetection(language="it", confidence=0.9)

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 4, dtype=np.float32)
    lang, det = lid.detect_language_for_segment(
        audio_array, SpeechSegment(0.5, 2.5), default="es", context_window_s=0.0
    )
    assert lang == "it"
    assert captured["len"] == 32000  # 2.0 s (>= MIN_LID_DURATION_S -> se analiza)


def test_detect_language_for_segment_corto_no_ejecuta_lid_y_hereda(monkeypatch):
    """Micro-chunk (<1.5 s): NO se ejecuta Whisper LID; hereda el idioma previo."""
    llamadas = []

    def boom(audio, **kwargs):  # si se llama, falla la prueba
        llamadas.append(len(audio))
        raise AssertionError("El LID no debe ejecutarse en micro-chunks")

    monkeypatch.setattr(lid, "detect_language", boom)
    audio_array = np.zeros(16000 * 5, dtype=np.float32)
    lang, det = lid.detect_language_for_segment(
        audio_array, SpeechSegment(2.0, 3.0), default="es",
        previous_language="it", previous_end=1.5,
    )
    assert llamadas == []             # sin inferencia de Whisper
    assert lang == "it"               # hereda el idioma del segmento anterior
    assert det.confidence == 0.0      # no hay detección real
    # Sin idioma previo -> idioma por defecto
    lang2, _ = lid.detect_language_for_segment(
        audio_array, SpeechSegment(2.0, 3.0), default="es"
    )
    assert lang2 == "es"


def test_detect_language_for_segment_expande_ventana_por_baja_confianza(monkeypatch):
    """Tramo >= 1.5 s con score bajo: añade ±1 s de audio circundante."""
    captured = {}

    def fake_detect(audio, **kwargs):
        captured.setdefault("lens", []).append(len(audio))
        return LanguageDetection(language="it", confidence=0.2)  # baja confianza

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 10, dtype=np.float32)
    lang, _ = lid.detect_language_for_segment(
        audio_array, SpeechSegment(1.0, 3.0), default="es"
    )
    assert lang == "es"               # 0.2 < umbral -> no confirma 'it'
    # 1ª llamada con el tramo (2.0 s); 2ª con ±1 s alrededor (0.0-4.0 s)
    assert captured["lens"] == [32000, 64000]


def test_detect_language_for_segment_hereda_idioma_anterior(monkeypatch):
    """Confianza baja persistente -> hereda el idioma del segmento anterior."""
    def fake_detect(audio, **kwargs):
        return LanguageDetection(language="fr", confidence=0.2)

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 5, dtype=np.float32)
    # Δt = 0.5 - 0.0 = 0.5 s < T_inertia: herencia activa
    lang, det = lid.detect_language_for_segment(
        audio_array, SpeechSegment(0.5, 2.5), default="es",
        previous_language="it", previous_end=0.0,
    )
    assert lang == "it"          # herencia del segmento anterior (Δt corto)
    assert not det.is_confident()
    # Sin anterior, cae al idioma por defecto
    lang2, _ = lid.detect_language_for_segment(
        audio_array, SpeechSegment(0.5, 2.5), default="es"
    )
    assert lang2 == "es"


def test_assign_languages_hereda_entre_segmentos(monkeypatch):
    """assign_languages encadena el idioma: los tramos de baja confianza heredan."""
    sequence = [("it", 0.95), ("en", 0.15), ("en", 0.15), ("de", 0.1), ("de", 0.1)]
    idx = {"n": 0}

    def fake_detect(audio, **kwargs):
        lang, conf = sequence[idx["n"]]
        idx["n"] += 1
        return LanguageDetection(language=lang, confidence=conf)

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 8, dtype=np.float32)
    segments = [
        SpeechSegment(0.5, 2.5).with_language("es"),
        SpeechSegment(3.0, 5.0).with_language("es"),
        SpeechSegment(5.5, 7.5).with_language("es"),
    ]
    annotated = lid.assign_languages(segments, audio_array)
    assert [s.language for s in annotated] == ["it", "it", "it"]


# ---------------------------------------------------------------------------
# 4) Restricción de idiomas candidatos (allowed_languages)
# ---------------------------------------------------------------------------
def test_allowed_languages_selecciona_mejor_candidato_restringido():
    """{'iso1':0.40, 'iso2':0.35, 'iso3':0.25} + allowed=['iso2','iso3'] -> iso2."""
    det = LanguageDetection(
        language="iso1", confidence=0.40,
        probabilities={"iso1": 0.40, "iso2": 0.35, "iso3": 0.25},
    )
    lang, ok = resolve_language(
        det, threshold=0.3, allowed_languages=["iso2", "iso3"]
    )
    assert lang == "iso2"      # 0.35 > 0.25 dentro del subconjunto permitido
    assert ok is True


def test_allowed_languages_none_o_vacio_usa_comportamiento_por_defecto():
    det = LanguageDetection(
        language="iso1", confidence=0.90,
        probabilities={"iso1": 0.90, "iso2": 0.08, "iso3": 0.02},
    )
    lang_none, _ = resolve_language(det, threshold=0.5, allowed_languages=None)
    lang_vacio, _ = resolve_language(det, threshold=0.5, allowed_languages=[])
    assert lang_none == lang_vacio == "iso1"   # evaluación completa


def test_allowed_languages_sin_candidatos_fallback_al_default():
    det = LanguageDetection(
        language="xx", confidence=0.0, probabilities={"xx": 1.0},
    )
    lang, ok = resolve_language(
        det, threshold=0.3, default="es", allowed_languages=["iso2", "iso3"]
    )
    assert (lang, ok) == ("es", False)


def test_allowed_languages_candidato_bajo_umbral_fallback():
    det = LanguageDetection(
        language="iso1", confidence=0.9,
        probabilities={"iso1": 0.9, "iso2": 0.1},
    )
    lang, ok = resolve_language(
        det, threshold=0.3, default="es", allowed_languages=["iso2"]
    )
    assert (lang, ok) == ("es", False)    # 0.1 < 0.3


def test_parse_allowed_languages_env_y_argumento(monkeypatch):
    monkeypatch.setenv(lid._ENV_ALLOWED_LANGUAGES, "es,it, EN ")
    assert lid.parse_allowed_languages() == ["es", "it", "en"]
    monkeypatch.setenv(lid._ENV_ALLOWED_LANGUAGES, "")
    assert lid.parse_allowed_languages() is None
    assert lid.parse_allowed_languages(" fr ,DE") == ["fr", "de"]


def test_allowed_languages_en_detect_language_for_segment(monkeypatch):
    """El flujo por segmento elige el candidato permitido con más score."""
    probabilities = {"es": 0.10, "it": 0.25, "fr": 0.60, "de": 0.05}

    def fake_detect(audio, **kwargs):
        return LanguageDetection(
            language="fr", confidence=0.60, probabilities=probabilities
        )

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 4, dtype=np.float32)
    lang, det = lid.detect_language_for_segment(
        audio_array, SpeechSegment(0.5, 2.5), default="es",
        allowed_languages=["es", "it"], threshold=0.2, context_window_s=0.0,
    )
    assert lang == "it"        # 0.25 es el máximo dentro de {"es", "it"}
    assert "it" in det.probabilities


# ---------------------------------------------------------------------------
# 5) Matriz de transición temporal (inercia según Δt)
# ---------------------------------------------------------------------------
def test_inercia_preserva_idioma_previo_si_dt_corto(monkeypatch):
    """Con Δt < T_inertia, el sesgo hace ganar al idioma del segmento anterior."""
    probabilities = {"es": 0.34, "it": 0.33, "fr": 0.33}   # ambiguo dentro de allowed

    def fake_detect(audio, **kwargs):
        return LanguageDetection(
            language="fr", confidence=0.33, probabilities=dict(probabilities)
        )

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 8, dtype=np.float32)
    # segmento actual [3.0, 5.0]; anterior terminaba en 2.5 -> Δt = 0.5 s < 2.0
    lang, det = lid.detect_language_for_segment(
        audio_array, SpeechSegment(3.0, 5.0), default="es",
        allowed_languages=["es", "it"], threshold=0.3,
        previous_language="it", previous_end=2.5, context_window_s=0.0,
    )
    assert lang == "it"        # 0.33 + INERTIA_BIAS > 0.34 (sin sesgo ganaría 'es')


def test_inercia_decae_a_cero_si_dt_largo(monkeypatch):
    """Con Δt > T_inertia la evaluación es neutra y vence el candidato superior."""
    probabilities = {"es": 0.34, "it": 0.33, "fr": 0.33}

    def fake_detect(audio, **kwargs):
        return LanguageDetection(
            language="fr", confidence=0.33, probabilities=dict(probabilities)
        )

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 8, dtype=np.float32)
    # Ventana explícita de 2 s y Δt = 3.0 s → fuera de ventana → evaluación neutra
    lang, _ = lid.detect_language_for_segment(
        audio_array, SpeechSegment(3.0, 5.0), default="es",
        allowed_languages=["es", "it"], threshold=0.3,
        previous_language="it", previous_end=0.0, context_window_s=0.0,
        t_inertia=2.0,
    )
    assert lang == "es"        # sin sesgo: 0.34 > 0.33


def test_bias_por_inercia_temporal_directo():
    det = LanguageDetection(
        language="es", confidence=0.31,
        probabilities={"es": 0.31, "it": 0.29},
    )
    sesgado = lid._bias_by_time_inertia(det, "it", delta_t=0.5)
    assert sesgado.probabilities["it"] == pytest.approx(0.29 + lid.INERTIA_BIAS)
    # Δt ≥ T_inertia (o sin idioma previo): probabilidades sin tocar
    neutro = lid._bias_by_time_inertia(det, "it", delta_t=lid.T_INERTIA_S + 1.0)
    assert neutro.probabilities == det.probabilities
    sin_previo = lid._bias_by_time_inertia(det, None, delta_t=0.5)
    assert sin_previo.probabilities == det.probabilities


def test_ventana_inercia_6s_incluye_pausas_conversacionales(monkeypatch):
    """Δt = 5.0 s (< 6.0 s por defecto) mantiene el turno y la Lengua B activa."""
    probabilities = {"es": 0.30, "it": 0.20, "fr": 0.50}

    def fake_detect(audio, **kwargs):
        return LanguageDetection(
            language="fr", confidence=0.50, probabilities=dict(probabilities)
        )

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 12, dtype=np.float32)
    lang, _ = lid.detect_language_for_segment(
        audio_array, SpeechSegment(8.0, 10.0), default="es",
        allowed_languages=["es", "it"], threshold=0.5,
        previous_language="it", previous_end=3.0, context_window_s=0.0,
    )
    # Δt = 5.0 < 6.0 → mismo turno: se hereda/mantiene 'it'
    assert lang == "it"


def test_histéresis_turno_mantiene_lengua_b_ante_evidencia_moderada(monkeypatch):
    """Cambiar it→es a mitad de turno con score 0.50 no supera 0.65 -> sigue it."""
    probabilities = {"es": 0.50, "it": 0.20, "fr": 0.30}

    def fake_detect(audio, **kwargs):
        return LanguageDetection(
            language="fr", confidence=0.30, probabilities=dict(probabilities)
        )

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 10, dtype=np.float32)
    lang, _ = lid.detect_language_for_segment(
        audio_array, SpeechSegment(4.0, 6.0), default="es",
        allowed_languages=["es", "it"], threshold=0.5,
        previous_language="it", previous_end=0.0, context_window_s=0.0,
    )
    # Δt = 4.0 < 6.0, candidato 'es' 0.50 < 0.50+0.15 -> histéresis conserva 'it'
    assert lang == "it"


def test_histéresis_turno_cede_con_evidencia_abrumadora(monkeypatch):
    """Cambio it→es a mitad de turno SOLO con doble confirmación acústica (≥0.95)."""
    probabilities = {"es": 0.97, "it": 0.01, "fr": 0.02}

    def fake_detect(audio, **kwargs):
        return LanguageDetection(
            language="fr", confidence=0.02, probabilities=dict(probabilities)
        )

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 10, dtype=np.float32)
    lang, _ = lid.detect_language_for_segment(
        audio_array, SpeechSegment(4.0, 6.0), default="es",
        allowed_languages=["es", "it"], threshold=0.5,
        previous_language="it", previous_end=0.0, context_window_s=0.0,
    )
    assert lang == "es"   # 0.97 >= max(0.5+0.15, 0.95)


def test_histéresis_turno_0_94_no_autoriza_salir_de_lengua_b(monkeypatch):
    """es @0.94 (pronunciación castellanizada) NO supera 0.95 -> se mantiene it."""
    probabilities = {"es": 0.94, "it": 0.03, "fr": 0.03}

    def fake_detect(audio, **kwargs):
        return LanguageDetection(
            language="fr", confidence=0.03, probabilities=dict(probabilities)
        )

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 10, dtype=np.float32)
    lang, _ = lid.detect_language_for_segment(
        audio_array, SpeechSegment(4.0, 6.0), default="es",
        allowed_languages=["es", "it"], threshold=0.5,
        previous_language="it", previous_end=0.0, context_window_s=0.0,
    )
    assert lang == "it"   # 0.94 < 0.95 -> doble confirmación acústica NO dada


def test_histéresis_solo_fuera_de_ventana_cambia_con_umbral_normal(monkeypatch):
    """Con Δt=7.0 s (> ventana 6.0 s) el español a 0.50 sí gana (turno nuevo)."""
    probabilities = {"es": 0.50, "it": 0.20, "fr": 0.30}

    def fake_detect(audio, **kwargs):
        return LanguageDetection(
            language="fr", confidence=0.30, probabilities=dict(probabilities)
        )

    monkeypatch.setattr(lid, "detect_language", fake_detect)
    audio_array = np.zeros(16000 * 12, dtype=np.float32)
    lang, _ = lid.detect_language_for_segment(
        audio_array, SpeechSegment(8.0, 10.0), default="es",
        allowed_languages=["es", "it"], threshold=0.5,
        previous_language="it", previous_end=1.0, context_window_s=0.0,
    )
    # Δt = 7.0 >= 6.0 → turno nuevo: 'es' con umbral normal (0.50 >= 0.5)
    assert lang == "es"


# ---------------------------------------------------------------------------
# 6) Clasificación de texto agnóstica (NLP: lingua -> langdetect)
# ---------------------------------------------------------------------------
NEEDS_LINGUA = pytest.mark.skipif(
    importlib.util.find_spec("lingua") is None,
    reason="Backend lingua-language-detector no instalado (code-switching requiere lingua)",
)


def test_detect_text_language_texto_claramente_lengua_b():
    assert lid.detect_text_language(
        "parliamo di questo progetto e delle sue potenzialità", "es", "it"
    ) == "it"


def test_detect_text_language_texto_claramente_es():
    resultado = lid.detect_text_language(
        "Hoy es un gran día y lo vamos a celebrar juntos mañana", "es", "it"
    )
    assert resultado in ("es", None)      # texto 100 % español nunca da 'it'
    assert resultado != "it"


@NEEDS_LINGUA
def test_detect_text_language_code_switching_a_lengua_b():
    """Code-switching 'Hoy es un grande onore...' -> la Lengua B gana al texto."""
    assert lid.detect_text_language(
        "Hoy es un grande onore para mí poder presentare questo progetto",
        "es", "it",
    ) == "it"


def test_detect_text_language_sin_texto_o_backend_ausente(monkeypatch):
    assert lid.detect_text_language("", "es", "it") is None
    assert lid.detect_text_language(None, "es", "it") is None
    assert lid.detect_text_language("   ", "es", "it") is None
    # Sin backend NLP -> None (el llamador mantiene el resultado del audio)
    monkeypatch.setattr(lid, "_text_detector_for", lambda *a, **k: None)
    assert lid.detect_text_language("frase cualquiera en es y it", "es", "it") is None


def test_detect_text_language_requiere_dos_idiomas_distintos():
    assert lid.detect_text_language("frase", "es") is None        # falta Lengua B
    assert lid.detect_text_language("frase", "es", "es") is None  # ambos iguales


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