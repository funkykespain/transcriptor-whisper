# -*- coding: utf-8 -*-
"""Tests del pipeline de pre-tratamiento de audio (`audio_preprocessing.py`).

Se generan audios sintéticos con NumPy + módulo estándar ``wave`` y se valida
el resultado con análisis FFT y con la propia medición EBU R128 de FFmpeg.
Requiere el binario ``ffmpeg`` (requisito del proyecto); si no está, los tests
se omiten automáticamente.
"""

import json
import shutil
import subprocess
import wave

import numpy as np
import pytest

from audio_preprocessing import (
    DEFAULT_SAMPLE_RATE,
    AudioPreprocessingError,
    convert_base_pcm,
    preprocess_audio_base,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="FFmpeg no está instalado en el sistema"
)


# ---------------------------------------------------------------------------
# Utilidades de test
# ---------------------------------------------------------------------------
def _write_wav(path, samples, sample_rate, n_channels=1, sampwidth=2):
    """Escribe `samples` (float -1..1, forma (n,) o (n, ch)) a un WAV PCM."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
    pcm = np.clip(samples, -1.0, 1.0) * (2 ** (8 * sampwidth - 1) - 1)
    pcm = pcm.astype("<i2" if sampwidth == 2 else "<i1").tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _read_wav(path):
    """Devuelve (muestras float mono, sample_rate, channels, sampwidth)."""
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        data = wf.readframes(wf.getnframes())
    dtype = {1: "<i1", 2: "<i2", 4: "<i4"}[sampwidth]
    samples = np.frombuffer(data, dtype=dtype).astype(np.float64)
    samples = samples / (2 ** (8 * sampwidth - 1))
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate, channels, sampwidth


def _measure_integrated_lufs(path):
    """Mide la loudness integrada (EBU R128) con el propio FFmpeg."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-i", str(path),
        "-af", "loudnorm=I=-18:print_format=json",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    idx = proc.stderr.find('"input_i"')
    start = proc.stderr.rfind("{", 0, idx)
    end = proc.stderr.find("}", idx) + 1
    return float(json.loads(proc.stderr[start:end])["input_i"])


def _db(magnitude):
    return 20.0 * np.log10(max(magnitude, 1e-12))


def _tone(duration_s, sample_rate, frequencies_amplitudes):
    """Mezcla de senos: [(frecuencia_hz, amplitud), ...] -> señal mono."""
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    signal = np.zeros_like(t)
    for freq, amp in frequencies_amplitudes:
        signal += amp * np.sin(2 * np.pi * freq * t)
    return signal


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_preprocess_audio_base_convierte_a_pcm_16k_mono_16_bits(tmp_path):
    """De un estéreo 44.1 kHz debe salir WAV mono 16 kHz, 16-bit (PCM s16le)."""
    secs = 2.0
    sr_in = 44100
    n = int(secs * sr_in)
    t = np.arange(n) / sr_in
    left = 0.3 * np.sin(2 * np.pi * 440 * t)
    right = 0.3 * np.sin(2 * np.pi * 880 * t)
    input_path = tmp_path / "input_stereo_44k.wav"
    _write_wav(input_path, np.stack([left, right], axis=1), sr_in, n_channels=2)

    output_path = tmp_path / "output.wav"
    returned = preprocess_audio_base(input_path, output_path)

    assert str(returned) == str(output_path.resolve())
    samples, sample_rate, channels, sampwidth = _read_wav(output_path)
    assert sample_rate == DEFAULT_SAMPLE_RATE == 16_000
    assert channels == 1
    assert sampwidth == 2  # 16-bit PCM
    assert samples.size > 0
    assert np.isfinite(samples).all()


def test_preprocess_audio_base_normaliza_a_18_lufs(tmp_path):
    """Un tono muy bajo (-26 dB) debe quedar normalizado a ~ -18 LUFS."""
    sr = 16000
    input_path = tmp_path / "quiet.wav"
    _write_wav(input_path, _tone(3.0, sr, [(1000.0, 0.05)]), sample_rate=sr)

    output_path = tmp_path / "normalized.wav"
    preprocess_audio_base(input_path, output_path)

    lufs = _measure_integrated_lufs(output_path)
    assert abs(lufs - (-18.0)) <= 1.0, f"LUFS medidos: {lufs:.2f} (objetivo -18)"


def test_preprocess_audio_base_paso_alto_80hz_atenda_subgraves(tmp_path):
    """Un tono a 20 Hz debe quedar claramente atenuado frente a uno a 1 kHz."""
    sr = DEFAULT_SAMPLE_RATE
    duration_s = 4.0
    input_path = tmp_path / "mixed.wav"
    _write_wav(
        input_path,
        _tone(duration_s, sr, [(20.0, 0.5), (1000.0, 0.25)]),
        sample_rate=sr,
    )
    output_path = tmp_path / "filtered.wav"
    preprocess_audio_base(input_path, output_path)

    samples, sample_rate, _, _ = _read_wav(output_path)
    n = samples.size
    # Hann + FFT: 20 Hz -> bin 80, 1 kHz -> bin 4000 (frecuencias exactas).
    windowed = samples * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    mag_20hz = spectrum[np.argmin(np.abs(freqs - 20.0))]
    mag_1khz = spectrum[np.argmin(np.abs(freqs - 1000.0))]
    assert _db(mag_1khz) - _db(mag_20hz) > 10.0, (
        f"1 kHz ({_db(mag_1khz):.1f} dB) debería dominar a 20 Hz "
        f"({_db(mag_20hz):.1f} dB) tras el paso alto de 80 Hz"
    )


def test_preprocess_audio_base_modo_single_pass_tambien_funciona(tmp_path):
    """El modo single pass (dinámico) debe producir el mismo formato de salida."""
    input_path = tmp_path / "in.wav"
    _write_wav(input_path, _tone(1.5, 16000, [(500.0, 0.2)]), sample_rate=16000)
    output_path = tmp_path / "single.wav"
    preprocess_audio_base(input_path, output_path, mode="single")

    _, sample_rate, channels, sampwidth = _read_wav(output_path)
    assert sample_rate == 16_000
    assert channels == 1
    assert sampwidth == 2


def test_preprocess_audio_base_valida_errores(tmp_path):
    """Casos de error: entrada inexistente y directorio de salida ausente."""
    output_path = tmp_path / "out.wav"
    with pytest.raises(AudioPreprocessingError):
        preprocess_audio_base(tmp_path / "no_existe.wav", output_path)

    input_path = tmp_path / "in.wav"
    _write_wav(input_path, _tone(0.5, 16000, [(440.0, 0.1)]), sample_rate=16000)
    with pytest.raises(AudioPreprocessingError):
        preprocess_audio_base(input_path, tmp_path / "subdir" / "out.wav")

    with pytest.raises(ValueError):
        preprocess_audio_base(input_path, output_path, mode="triple_pass")


def test_preprocess_audio_base_error_si_ffmpeg_ausente(tmp_path, monkeypatch):
    """Sin ffmpeg en el PATH debe lanzar AudioPreprocessingError."""
    import audio_preprocessing as ap

    input_path = tmp_path / "in.wav"
    _write_wav(input_path, _tone(0.5, 16000, [(440.0, 0.1)]), sample_rate=16000)

    monkeypatch.setattr(ap.shutil, "which", lambda name: None)
    with pytest.raises(AudioPreprocessingError):
        ap.preprocess_audio_base(input_path, tmp_path / "out.wav")


# ---------------------------------------------------------------------------
# Puerta de ruido (agate) y referencia de energía SIN loudnorm
# ---------------------------------------------------------------------------
def _rms_db_region(path, start_s, end_s):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        data = wf.readframes(wf.getnframes())
    samples = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    region = samples[int(start_s * sr):int(end_s * sr)]
    return 20.0 * np.log10(np.sqrt(np.mean(region ** 2)) + 1e-12)


def _tono_puro(amp, freq, duration_s, sample_rate=16000):
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    return amp * np.sin(2 * np.pi * freq * t)


def test_puerta_de_ruido_atenda_fuga_sin_tocar_la_voz(tmp_path):
    """El expansor agate reduce el contenido bajo -32 dB y deja la voz intacta."""
    sr = 16000
    loud = _tono_puro(0.5, 500.0, 1.0, sr)     # voz  ≈ -9 dBFS  (> umbral)
    bleed = _tono_puro(0.02, 550.0, 1.0, sr)   # fuga ≈ -37 dBFS (< umbral)
    src = tmp_path / "in.wav"
    _write_wav(src, np.concatenate([loud, bleed]), sample_rate=sr)

    con_gate = tmp_path / "con_gate.wav"
    sin_gate = tmp_path / "sin_gate.wav"
    convert_base_pcm(src, con_gate, noise_gate=True)
    convert_base_pcm(src, sin_gate, noise_gate=False)

    # La voz no se toca (diferencia < 0.5 dB)
    assert abs(_rms_db_region(con_gate, 0.0, 1.0) - _rms_db_region(sin_gate, 0.0, 1.0)) < 0.5
    # La fuga bajo el umbral se atenúa >= 10 dB en estado estacionario
    # (medido en los últimos 500 ms del tramo, tras el transitorio de release)
    bleed_sin = _rms_db_region(sin_gate, 1.5, 2.0)
    bleed_con = _rms_db_region(con_gate, 1.5, 2.0)
    assert bleed_sin - bleed_con >= 10.0


def test_convert_base_pcm_es_pcm_16k_mono_16_bit_sin_loudnorm(tmp_path):
    """La referencia de energía es PCM 16 kHz/mono/16-bit (sin compresión)."""
    sr = 44100
    t = np.arange(int(2 * sr)) / sr
    stereo = np.stack([
        0.5 * np.sin(2 * np.pi * 440 * t),
        0.5 * np.sin(2 * np.pi * 660 * t),
    ], axis=1)
    src = tmp_path / "stereo44k.wav"
    _write_wav(src, stereo, sample_rate=sr, n_channels=2)

    out = tmp_path / "energia.wav"
    returned = convert_base_pcm(src, out)
    assert str(returned) == str(out.resolve())
    with wave.open(str(out), "rb") as wf:
        assert wf.getframerate() == 16_000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2