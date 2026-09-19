# -*- coding: utf-8 -*-
"""Tests del detector de voz (`audio_vad.py`).

Cubren: (1) la lógica pura de padding+fusión de 400 ms, (2) la validación del
formato normalizado (16 kHz mono 16-bit), (3) el pipeline end-to-end con el
modelo Silero VAD real (se omiten si no hay red/dependencias) y (4) la
exportación de chunks. La integración Fase 1 -> Fase 2 se ejerce en el test
de audio real pasando el fichero por ``preprocess_audio_base``.
"""

import os
import urllib.request
import wave

import numpy as np
import pytest

import audio_vad as vad
from audio_preprocessing import preprocess_audio_base

MODEL_READY = None


def _model_is_ready() -> bool:
    global MODEL_READY
    if MODEL_READY is None:
        try:
            vad.get_vad_model()
            MODEL_READY = True
        except Exception:  # noqa: BLE001 - sin red/sin deps: se omiten esos tests
            MODEL_READY = False
    return MODEL_READY


requires_model = pytest.mark.skipif(
    not _model_is_ready(), reason="Modelo Silero VAD no disponible (sin red o sin dependencias)"
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


def _read_wav(path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        data = wf.readframes(wf.getnframes())
    samples = np.frombuffer(data, dtype="<i2" if sw == 2 else "<i1").astype(np.float64)
    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1)
    return samples, sr, ch, sw


def _tone(duration_s, sample_rate=16000, freq=440.0, amp=0.3):
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    return amp * np.sin(2 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# 1) Lógica pura: padding de 400 ms antes/después + fusión + recorte de bordes
# ---------------------------------------------------------------------------
def test_pad_and_merge_aplica_400ms_a_ambos_lados():
    result = vad._pad_and_merge([(1.0, 2.0)], duration_s=10.0, pad_ms=400)
    assert result == [(0.6, 2.4)]


def test_pad_and_merge_fusiona_tramos_solapados():
    result = vad._pad_and_merge([(1.0, 2.0), (2.3, 3.0)], duration_s=10.0, pad_ms=400)
    assert result == [(0.6, 3.4)]  # 2.3-0.4=1.9 <= 2.0+0.4=2.4 => fusionado


def test_pad_and_merge_no_fusiona_tramos_lejanos():
    result = vad._pad_and_merge([(0.0, 0.2), (2.0, 2.2)], duration_s=10.0, pad_ms=400)
    assert np.allclose(result, [(0.0, 0.6), (1.6, 2.6)])


def test_pad_and_merge_recorta_en_bordes_del_audio():
    result = vad._pad_and_merge([(0.1, 0.5), (9.5, 9.8)], duration_s=10.0, pad_ms=1000)
    assert result == [(0.0, 1.5), (8.5, 10.0)]


def test_pad_and_merge_rechaza_padding_negativo():
    with pytest.raises(ValueError):
        vad._pad_and_merge([(0.0, 1.0)], duration_s=10.0, pad_ms=-100)


# ---------------------------------------------------------------------------
# 2) Estructura de datos y validación de formato normalizado
# ---------------------------------------------------------------------------
def test_speech_segment_to_dict_y_to_ms():
    seg = vad.SpeechSegment(start=12.345, end=13.5)
    d = seg.to_dict()
    assert d["start"] == 12.345
    assert d["start_ms"] == 12345
    assert d["end_ms"] == 13500
    assert seg.to_ms() == (12345, 13500)
    assert abs(seg.duration - 1.155) < 1e-9


def test_speech_segment_invalido():
    with pytest.raises(ValueError):
        vad.SpeechSegment(start=2.0, end=1.0)


def test_detect_rechaza_formato_no_normalizado(tmp_path):
    # Frecuencia de muestreo incorrecta (44.1 kHz)
    bad_sr = tmp_path / "bad_sr.wav"
    _write_wav(bad_sr, _tone(1.0), sample_rate=44100)
    with pytest.raises(vad.VadError):
        vad.detect_speech_segments(bad_sr)

    # Estéreo
    bad_ch = tmp_path / "bad_ch.wav"
    _write_wav(bad_ch, _tone(1.0), n_channels=2)
    with pytest.raises(vad.VadError):
        vad.detect_speech_segments(bad_ch)

    # Inexistente
    with pytest.raises(vad.VadError):
        vad.detect_speech_segments(tmp_path / "no_existe.wav")


def test_detect_audio_muy_corto_devuelve_vacio(tmp_path):
    short = tmp_path / "short.wav"
    _write_wav(short, np.zeros(100))
    assert vad.detect_speech_segments(short) == []


# ---------------------------------------------------------------------------
# 3) Pipeline end-to-end con el modelo real (Silero VAD)
# ---------------------------------------------------------------------------
@requires_model
def test_detect_speech_segments_sintetico_end_to_end(tmp_path):
    audio = tmp_path / "sintetico.wav"
    _write_wav(audio, _tone(3.0))
    segments = vad.detect_speech_segments(audio)
    assert isinstance(segments, list)
    for segment in segments:
        assert 0.0 <= segment.start < segment.end <= 3.0
        # El padding de 400 ms queda aplicado: cada tramo mide >= 0.8 s
        assert segment.duration >= 0.8 - 1e-6


@requires_model
def test_detect_speech_segments_audio_real_en_fase1_fase2(tmp_path):
    """Audio real de referencia de Silero: detecta habla y respeta los límites."""
    src = tmp_path / "en.wav"
    try:
        urllib.request.urlretrieve(
            "https://models.silero.ai/vad_models/en.wav", str(src)
        )
    except Exception as exc:  # noqa: BLE001 - sin red: omitimos
        pytest.skip(f"No se pudo descargar el ejemplo oficial: {exc}")

    normalized = tmp_path / "en_normalized.wav"
    preprocess_audio_base(src, normalized)  # Fase 1 (formato objetivo)
    segments = vad.detect_speech_segments(normalized)  # Fase 2
    assert len(segments) >= 1, "El ejemplo oficial contiene habla y no se detectó nada"
    for segment in segments:
        assert 0.0 <= segment.start < segment.end


# ---------------------------------------------------------------------------
# 4) Exportación / recorte de chunks
# ---------------------------------------------------------------------------
def test_export_speech_chunks_recorta_wav(tmp_path):
    duration_s = 4.0
    audio = tmp_path / "input.wav"
    _write_wav(audio, _tone(duration_s))

    segments = [vad.SpeechSegment(start=0.5, end=1.5)]
    out_dir = tmp_path / "chunks"
    paths = vad.export_speech_chunks(audio, segments, out_dir)

    assert len(paths) == 1
    chunk = paths[0]
    assert os.path.isfile(chunk)
    assert os.path.basename(chunk) == "chunk_0001_000500_001500.wav"

    samples, sr, ch, sw = _read_wav(chunk)
    expected = int(16000 * 1.0)
    assert sr == 16000
    assert ch == 1
    assert sw == 2
    assert len(samples) == pytest.approx(expected, abs=1)


def test_export_speech_chunks_crea_directorio_si_no_existe(tmp_path):
    audio = tmp_path / "input.wav"
    _write_wav(audio, _tone(2.0))
    paths = vad.export_speech_chunks(
        audio, [vad.SpeechSegment(0.0, 0.5), vad.SpeechSegment(1.0, 1.5)],
        tmp_path / "nuevo" / "subdir",
    )
    assert len(paths) == 2
    assert all(os.path.isfile(p) for p in paths)


def test_cut_speech_chunk_recorta_en_muestra_exacta(tmp_path):
    audio = tmp_path / "input.wav"
    _write_wav(audio, _tone(2.0))
    out = tmp_path / "un_chunk.wav"
    returned = vad.cut_speech_chunk(audio, vad.SpeechSegment(0.25, 1.75), out)
    assert returned == str(out.resolve())
    samples, _, _, _ = _read_wav(out)
    assert len(samples) == int(16000 * 1.5)