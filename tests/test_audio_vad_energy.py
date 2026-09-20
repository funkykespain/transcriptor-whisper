# -*- coding: utf-8 -*-
"""Tests del filtro de energía relativa anti "headphone bleed" (`audio_vad.py`).

Simula un audio con un hablante principal potente y una voz de fondo a -15 dB
(sin relación con la ganancia ni el idioma) y verifica que
``filter_segments_by_relative_energy`` descarta correctamente la voz lejana.
"""

import os
import wave

import numpy as np
import pytest

import audio_vad as vad
from audio_vad import (
    DEFAULT_ENERGY_MARGIN_DB,
    filter_segments_by_relative_energy,
)
from audio_vad import SpeechSegment

SAMPLE_RATE = 16_000

# Señales: tono puro de 500 Hz (periodos enteros en 1 s -> RMS exacto)
LOUD_AMP = 0.5
QUIET_DB_REL = 15.0                              # voz de fondo atenuada -15 dB
QUIET_AMP = LOUD_AMP * 10 ** (-QUIET_DB_REL / 20)

LOUD_RMS_DB = 20.0 * np.log10((LOUD_AMP / np.sqrt(2)) / 1.0)  # ≈ -9.03 dBFS
QUIET_RMS_DB = 20.0 * np.log10((QUIET_AMP / np.sqrt(2)) / 1.0)


def _build_audio_with_bleed(path, scale=1.0):
    """Audio de 1 s por bloque (tono + 0.2 s de silencio): 5 potentes + 3 fondo."""
    frequencies = {"loud": 500.0, "quiet": 500.0}
    amplitudes = {"loud": LOUD_AMP * scale, "quiet": QUIET_AMP * scale}
    order = ["loud", "loud", "quiet", "loud", "quiet", "loud", "quiet", "loud"]

    samples = []
    segments, expected_kept = [], []
    block_ms = 1200  # 1.0 s tono + 0.2 s silencio
    for index, kind in enumerate(order):
        start_ms = index * block_ms
        start_s, end_s = start_ms / 1000.0, (start_ms + 1000) / 1000.0
        segments.append(SpeechSegment(start_s, end_s))
        t = np.arange(int(1.0 * SAMPLE_RATE)) / SAMPLE_RATE
        tone = amplitudes[kind] * np.sin(2 * np.pi * frequencies[kind] * t)
        samples.append(tone)
        samples.append(np.zeros(int(0.2 * SAMPLE_RATE)))
        if kind == "loud":
            expected_kept.append(index)

    audio = np.concatenate(samples).astype(np.float32)
    pcm = np.clip(audio, -1.0, 1.0) * 32767
    pcm = pcm.astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return segments, expected_kept


# ---------------------------------------------------------------------------
# 1) Filtro por energía relativa
# ---------------------------------------------------------------------------
def test_filtro_elimina_voz_de_fondo_a_15db(tmp_path):
    audio = tmp_path / "con_fuga.wav"
    segments, expected = _build_audio_with_bleed(audio)

    kept = filter_segments_by_relative_energy(audio, segments, energy_margin_db=12.0)

    assert len(kept) == len(expected) == 5
    assert [segments.index(s) for s in kept] == expected
    for idx in range(len(segments)):
        if idx in expected:
            assert segments[idx] in kept


def test_filtro_es_agnostico_a_la_ganancia(tmp_path):
    """Escalar todo el audio -20 dB no cambia la decisión (energía relativa)."""
    audio = tmp_path / "con_fuga_atenuado.wav"
    segments, expected = _build_audio_with_bleed(audio, scale=0.1)

    kept = filter_segments_by_relative_energy(audio, segments, energy_margin_db=12.0)
    assert [segments.index(s) for s in kept] == expected


def test_margen_configurable_por_argumento(tmp_path):
    """Margen 20 dB conserva la voz de fondo a -15 dB; margen 12 la elimina."""
    audio = tmp_path / "con_fuga.wav"
    segments, expected = _build_audio_with_bleed(audio)

    kept_exigente = filter_segments_by_relative_energy(audio, segments, energy_margin_db=12.0)
    kept_permisivo = filter_segments_by_relative_energy(audio, segments, energy_margin_db=20.0)

    assert len(kept_exigente) == len(expected)
    assert len(kept_permisivo) == len(segments)   # -15 dB está dentro del margen de 20


def test_margen_configurable_por_env(monkeypatch, tmp_path):
    """La variable de entorno AUDIO_VAD_ENERGY_MARGIN_DB configura el filtro."""
    audio = tmp_path / "con_fuga.wav"
    segments, expected = _build_audio_with_bleed(audio)

    monkeypatch.setenv(vad._ENV_ENERGY_MARGIN_DB, "14")     # margen 14 dB
    kept_env = filter_segments_by_relative_energy(audio, segments)
    assert len(kept_env) == len(expected)

    monkeypatch.setenv(vad._ENV_ENERGY_MARGIN_DB, "40")     # margen enorme
    kept_laxos = filter_segments_by_relative_energy(audio, segments)
    assert len(kept_laxos) == len(segments)


def test_filtro_segmentos_vacios(tmp_path):
    audio = tmp_path / "nada.wav"
    segments, _ = _build_audio_with_bleed(audio)
    assert filter_segments_by_relative_energy(audio, []) == []


def test_filtro_con_unica_voz_no_descarta(tmp_path):
    audio = tmp_path / "solo_principal.wav"
    segments, _ = _build_audio_with_bleed(audio)
    only_loud = [s for i, s in enumerate(segments) if i in (0, 1, 3, 5, 7)]
    kept = filter_segments_by_relative_energy(audio, only_loud, energy_margin_db=12.0)
    assert len(kept) == len(only_loud)


def test_percentil_invalido_rechazado(tmp_path):
    audio = tmp_path / "x.wav"
    segments, _ = _build_audio_with_bleed(audio)
    with pytest.raises(ValueError):
        filter_segments_by_relative_energy(audio, segments, energy_percentile=110)


# ---------------------------------------------------------------------------
# 2) Medición de energía (RMS dBFS)
# ---------------------------------------------------------------------------
def test_compute_segments_energy_db_valores_esperados(tmp_path):
    audio = tmp_path / "con_fuga.wav"
    segments, _ = _build_audio_with_bleed(audio)

    energies = vad.compute_segments_energy_db(audio, segments)
    # Índices potentes: 0,1,3,5,7 | fondo: 2,4,6
    loud_energies = [energies[i] for i in (0, 1, 3, 5, 7)]
    quiet_energies = [energies[i] for i in (2, 4, 6)]
    assert all(abs(e - LOUD_RMS_DB) < 0.3 for e in loud_energies)
    assert all(abs(e - QUIET_RMS_DB) < 0.3 for e in quiet_energies)
    # La voz de fondo es ~-15 dB respecto a la principal
    diff = np.mean(loud_energies) - np.mean(quiet_energies)
    assert abs(diff - QUIET_DB_REL) < 0.6


# ---------------------------------------------------------------------------
# 3) Parámetros VAD por defecto (anti fuga) y validación
# ---------------------------------------------------------------------------
def test_defaults_vad_antifuga():
    assert vad.DEFAULT_THRESHOLD == 0.65
    assert vad.DEFAULT_MIN_SPEECH_DURATION_MS == 600
    assert DEFAULT_ENERGY_MARGIN_DB == 12.0


def test_filtro_rechaza_archivo_inexistente(tmp_path):
    audio = tmp_path / "con_fuga.wav"
    segments, _ = _build_audio_with_bleed(audio)
    with pytest.raises(vad.VadError):
        filter_segments_by_relative_energy(tmp_path / "no_existe.wav", segments)


def test_filtro_usa_referencia_energia_sin_loudnorm(tmp_path):
    """Flujo real (app.py): el RMS se mide sobre convert_base_pcm (sin loudnorm)."""
    from audio_preprocessing import convert_base_pcm

    src = tmp_path / "con_fuga.wav"
    segments, expected = _build_audio_with_bleed(src)

    # La referencia de energía es la Fase 1 SIN compresión loudnorm
    energy = tmp_path / "energia_raw.wav"
    convert_base_pcm(src, energy)

    kept = filter_segments_by_relative_energy(energy, segments, energy_margin_db=12.0)
    assert [segments.index(s) for s in kept] == expected

    # Esa referencia preserva la separación real (voz -15 dB vs fondo)
    energies = vad.compute_segments_energy_db(energy, segments)
    loud_mean = np.mean([energies[i] for i in (0, 1, 3, 5, 7)])
    quiet_mean = np.mean([energies[i] for i in (2, 4, 6)])
    assert abs((loud_mean - quiet_mean) - QUIET_DB_REL) < 1.0