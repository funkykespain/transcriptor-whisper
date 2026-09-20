# -*- coding: utf-8 -*-
"""Pipeline de pre-tratamiento de audio para la entrada a ASR (Whisper).

Convierte cualquier archivo de audio a PCM WAV (16 kHz, mono, 16-bit) y aplica
dos mejoras acústicas estándar antes de transcribir:

1. **Filtro paso alto a 80 Hz** ....... elimina retumbe de baja frecuencia
                                        (rumble, HVAC, pop del micrófono).
2. **Normalización EBU R128 (-18 LUFS)** alinea el nivel de escucha de cualquier
                                        fuente para que el ASR reciba un
                                        volumen consistente.

Sintaxis FFmpeg de referencia (context7 / https://ffmpeg.org/ffmpeg-all.html)::

    # Cadena de filtros core (aplicada sobre el audio ya en formato objetivo):
    ffmpeg -i input.wav \
        -af "loudnorm=I=-18:LRA=11:TP=-1.5:measured_I=...:measured_TP=...\
:measured_LRA=...:measured_thresh=...:linear=true, aresample=16000" \
        -c:a pcm_s16le -f wav output.wav

``loudnorm`` (EBU R128) soporta dos modos:

* **single pass** (dinámico): el que FFmpeg usa por defecto cuando no se
  suministran valores medidos. Rápido, pensado para flujos en tiempo real,
  pero no garantiza un valor integrado exacto de -18 LUFS.
* **two pass** (lineal): requiere una primera pasada de medición
  (``print_format=json``) y reutiliza los valores ``measured_*`` con
  ``linear=true``. Es el modo **óptimo para archivos** porque clava el objetivo
  de -18 LUFS de forma determinista. Es el modo por defecto de este pipeline.

Para garantizar determinismo, el pipeline se ejecuta en **3 subprocesos**:

1. **Conversión + paso alto**: `ffmpeg -i entrada -af highpass=f=80
   -ar 16000 -ac 1 -c:a pcm_s16le -f wav` (sin ``loudnorm``: el re-muestreo y
   el downmix se aplican de forma estable, sin que el optimizador de FFmpeg
   reordene nada).
2. **Medición** (solo en modo *two pass*): ``loudnorm=...:print_format=json``
   sobre el archivo intermedio para obtener los valores ``measured_*``.
3. **Normalización**: ``loudnorm ... :linear=true`` seguido de un
   ``aresample`` **explícito al final de la cadena**. Poner únicamente ``-ar``
   como opción de salida es un error silencioso: FFmpeg inserta el re-muestreo
   ANTES de ``loudnorm`` (que trabaja internamente a 192 kHz) y el resultado ya
   no alcanza los -18 LUFS objetivo (medido en la práctica: -24 LUFS en vez de
   -18 para un estéreo 44.1 kHz).

La ruta del binario se resuelve con ``shutil.which("ffmpeg")`` (requisito del
proyecto), sin dependencias Python adicionales. Se evita así añadir
ffmpeg-python al proyecto ya que la app ya depende del binario de sistema.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Especificaciones del pipeline (valores por defecto del estándar Whisper/ASR)
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE: int = 16_000           # 16 kHz
DEFAULT_CHANNELS: int = 1                   # Mono
DEFAULT_SAMPLE_FORMAT: str = "pcm_s16le"    # PCM 16-bit signed little-endian
DEFAULT_HIGHPASS_HZ: float = 80.0           # Filtro paso alto (-3 dB a 80 Hz)
DEFAULT_TARGET_LUFS: float = -18.0          # EBU R128 integrated loudness
DEFAULT_LRA_TARGET: float = 11.0            # Loudness Range (voz/diálogo)
DEFAULT_TRUE_PEAK_DB: float = -1.5          # True peak pico máximo seguro ASR

# --- Puerta de ruido / expansor suave (agate) ANTES de loudnorm -------------
# Reduce el ruido de fondo y las fugas de bajo nivel (headphone bleed) antes
# de que loudnorm los 'suba'. threshold en dB (FFmpeg acepta sufijo dB),
# range 0..1 (reducción máxima, 0.12 ≈ -18 dB nominal / ~-9 dB medidos 1:1).
DEFAULT_GATE_THRESHOLD_DB: float = -32.0
DEFAULT_GATE_RANGE: float = 0.12
DEFAULT_GATE_RATIO: float = 4.0
DEFAULT_GATE_ATTACK_MS: float = 20.0
DEFAULT_GATE_RELEASE_MS: float = 250.0

_MODE_SINGLE_PASS = "single"
_MODE_TWO_PASS = "two_pass"
_VALID_MODES = (_MODE_SINGLE_PASS, _MODE_TWO_PASS)

# Claves del JSON de medición de loudnorm -> parámetros de la pasada lineal.
_MEASURED_KEY_MAP = {
    "input_i": "measured_I",
    "input_lra": "measured_LRA",
    "input_tp": "measured_TP",
    "input_thresh": "measured_thresh",
}

PathLike = Union[str, os.PathLike]


class AudioPreprocessingError(RuntimeError):
    """Error controlado del pipeline de preprocesado de audio."""


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str:
    """Devuelve la ruta de ``ffmpeg`` o lanza :class:`AudioPreprocessingError`."""
    binary = shutil.which("ffmpeg")
    if not binary:
        raise AudioPreprocessingError(
            "No se encontró 'ffmpeg' en el PATH. Es un requisito del proyecto "
            "(apt install ffmpeg / brew install ffmpeg)."
        )
    return binary


def _loudnorm_filter(
    target_lufs: float,
    lra: float,
    true_peak_db: float,
    measured: Optional[Dict[str, float]] = None,
    linear: bool = False,
) -> str:
    """Construye el filtro ``loudnorm`` (EBU R128) con la sintaxis FFmpeg."""
    parts = [f"I={target_lufs:.1f}", f"LRA={lra:g}", f"TP={true_peak_db:g}"]
    if measured:
        for name, value in measured.items():
            parts.append(f"{name}={value}")
    if linear:
        parts.append("linear=true")
    return "loudnorm=" + ":".join(parts)


def _noise_gate_filter() -> str:
    """Filtro ``agate`` (expansor/puerta suave) contra fuga de auriculares.

    Se aplica ANTES de loudnorm para atenuar el contenido bajo el umbral
    (ruido de fondo, voz lejana del examinador) sin afectar a la voz cercana:
    medido con FFmpeg 8.0.1, un tono a -37 dBFS se reduce ~9.5 dB mientras que
    la señales sobre -32 dBFS pasan intactas (0.0 dB).
    """
    return (
        f"agate=threshold={DEFAULT_GATE_THRESHOLD_DB:.0f}dB"
        f":range={DEFAULT_GATE_RANGE:g}"
        f":ratio={DEFAULT_GATE_RATIO:g}"
        f":attack={DEFAULT_GATE_ATTACK_MS:g}"
        f":release={DEFAULT_GATE_RELEASE_MS:g}"
    )


def _parse_loudnorm_stats(stderr: str) -> Dict[str, float]:
    """Extrae el JSON de medición de ``loudnorm=print_format=json`` (stderr)."""
    idx = stderr.find('"input_i"')
    if idx == -1:
        raise AudioPreprocessingError(
            "No se obtuvieron estadísticas loudnorm de la pasada de medición."
        )
    start = stderr.rfind("{", 0, idx)
    if start == -1:
        raise AudioPreprocessingError("JSON de loudnorm no encontrado en stderr.")
    end = stderr.find("}", idx) + 1
    if end <= 0:
        raise AudioPreprocessingError("JSON de loudnorm truncado en stderr.")
    try:
        stats = json.loads(stderr[start:end])
    except json.JSONDecodeError as exc:
        raise AudioPreprocessingError(f"Estadísticas loudnorm inválidas: {exc}") from exc
    return {target: float(stats[source]) for source, target in _MEASURED_KEY_MAP.items()}


def _convert_command(
    binary: str,
    input_path: PathLike,
    intermediate_path: PathLike,
    highpass_hz: float,
    sample_rate: int,
    channels: int,
    noise_gate: bool = True,
) -> list:
    """Etapa 1: convierte a PCM (``sample_rate``/``channels``/16-bit) y filtra.

    Cadena: ``highpass`` (+ puerta de ruido ``agate``) y después re-muestreo y
    downmix. Sin ``loudnorm`` de por medio, ``-ar``/``-ac`` se aplican de forma
    estable tras el filtro, sin reordenamientos del optimizador de FFmpeg.
    """
    af = f"highpass=f={highpass_hz:g}"
    if noise_gate:
        af += f",{_noise_gate_filter()}"
    return [
        binary, "-hide_banner", "-nostdin", "-y",
        "-i", str(input_path),
        "-af", af,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", DEFAULT_SAMPLE_FORMAT,
        "-f", "wav",
        str(intermediate_path),
    ]


def _measure_command(
    binary: str,
    intermediate_path: PathLike,
    target_lufs: float,
    lra: float,
    true_peak_db: float,
) -> list:
    """Etapa 2 (solo *two pass*): mide la loudness del archivo intermedio."""
    af = f"{_loudnorm_filter(target_lufs, lra, true_peak_db)}:print_format=json"
    return [
        binary, "-hide_banner", "-nostdin", "-y",
        "-i", str(intermediate_path),
        "-af", af,
        "-f", "null", "-",
    ]


def _normalize_command(
    binary: str,
    intermediate_path: PathLike,
    output_path: PathLike,
    target_lufs: float,
    lra: float,
    true_peak_db: float,
    measured: Optional[Dict[str, float]],
    sample_rate: int,
    channels: int,
) -> list:
    """Etapa 3: aplica ``loudnorm`` (EBU R128) y fija la tasa final.

    Importante: ``aresample`` va **explícito al final de la cadena** porque
    ``loudnorm`` trabaja internamente a 192 kHz; así el re-muestreo a
    ``sample_rate`` ocurre SIEMPRE después de normalizar (poner solo
    ``-ar`` como opción de salida permite que FFmpeg lo inserte antes de
    ``loudnorm`` y rompa la exactitud de los -18 LUFS).
    """
    loudnorm = _loudnorm_filter(
        target_lufs, lra, true_peak_db, measured=measured, linear=measured is not None
    )
    af = f"{loudnorm},aresample={sample_rate}"
    return [
        binary, "-hide_banner", "-nostdin", "-y",
        "-i", str(intermediate_path),
        "-af", af,
        "-ac", str(channels),
        "-c:a", DEFAULT_SAMPLE_FORMAT,
        "-f", "wav",
        str(output_path),
    ]


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def preprocess_audio_base(
    input_path: PathLike,
    output_path: PathLike,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    highpass_hz: float = DEFAULT_HIGHPASS_HZ,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    lra: float = DEFAULT_LRA_TARGET,
    true_peak_db: float = DEFAULT_TRUE_PEAK_DB,
    mode: str = _MODE_TWO_PASS,
) -> str:
    """Convierte ``input_path`` a PCM WAV 16 kHz / mono / 16-bit preprocesado.

    El pipeline aplica, en orden: filtro paso alto a ``highpass_hz`` (80 Hz por
    defecto) y normalización EBU R128 (``loudnorm``) a ``target_lufs``
    (-18 LUFS por defecto). La salida se escribe en ``output_path`` (WAV PCM).

    Args:
        input_path: Archivo de audio origen (wav, mp3, m4a, ogg, aac...).
        output_path: Ruta de salida, debe terminar en ``.wav``.
        sample_rate: Tasa de muestreo de salida (16 kHz por defecto, estándar ASR).
        channels: Canales de salida (1 = mono por defecto).
        highpass_hz: Frecuencia de corte del filtro paso alto.
        target_lufs: Loudness integrada objetivo (EBU R128), -18 LUFS por defecto.
        lra: Loudness Range objetivo (11 LU por defecto, recomendado voz).
        true_peak_db: True peak máximo (dBTP), -1.5 dB por defecto.
        mode: ``"two_pass"`` (recomendado para archivos: tres subprocesos con
            normalización lineal exacta a ``target_lufs``) o ``"single"``
            (normalización dinámica en una sola pasada de loudnorm, más rápida;
            no garantiza el valor exacto de target).

    Returns:
        La ruta absoluta de ``output_path`` generado.

    Raises:
        AudioPreprocessingError: Si ffmpeg falta, la entrada no existe, el
            directorio de salida no existe o FFmpeg falla.
        ValueError: Si ``mode`` no es un valor válido.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"mode debe ser {_MODE_TWO_PASS!r} o {_MODE_SINGLE_PASS!r}, se recibió {mode!r}."
        )
    if not os.path.isfile(input_path):
        raise AudioPreprocessingError(f"El archivo de entrada no existe: {input_path}")
    absolute_output = os.path.abspath(output_path)
    out_dir = os.path.dirname(absolute_output)
    if not os.path.isdir(out_dir):
        raise AudioPreprocessingError(
            f"El directorio de salida no existe: {out_dir}"
        )

    binary = _find_ffmpeg()

    # Archivo intermedio: audio ya en formato PCM objetivo (16k/mono/16-bit)
    # y con el paso alto aplicado. Sobre él se miden y aplican los -18 LUFS.
    fd, intermediate_path = tempfile.mkstemp(
        prefix=".preprocess_audio_base_", suffix=".wav", dir=out_dir
    )
    os.close(fd)
    try:
        proc = subprocess.run(
            _convert_command(binary, input_path, intermediate_path, highpass_hz, sample_rate, channels),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise AudioPreprocessingError(
                f"FFmpeg falló al convertir '{input_path}':\n{proc.stderr[-2000:]}"
            )

        measured = None
        if mode == _MODE_TWO_PASS:
            # Pasada de medición: loudness integrada real del audio ya filtrado.
            proc = subprocess.run(
                _measure_command(binary, intermediate_path, target_lufs, lra, true_peak_db),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise AudioPreprocessingError(
                    f"FFmpeg no pudo medir la loudness:\n{proc.stderr[-2000:]}"
                )
            measured = _parse_loudnorm_stats(proc.stderr)
            logger.debug("Loudness medida (EBU R128): %s", measured)

        # Pasada de normalización: lineal (two pass) o dinámica (single pass).
        proc = subprocess.run(
            _normalize_command(
                binary, intermediate_path, absolute_output, target_lufs, lra,
                true_peak_db, measured, sample_rate, channels,
            ),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise AudioPreprocessingError(
                f"FFmpeg falló al normalizar el audio:\n{proc.stderr[-2000:]}"
            )
    finally:
        try:
            os.remove(intermediate_path)
        except OSError:
            pass

    if not os.path.isfile(absolute_output):
        raise AudioPreprocessingError(
            f"FFmpeg terminó pero no generó la salida esperada: {absolute_output}"
        )
    return absolute_output


def convert_base_pcm(
    input_path: PathLike,
    output_path: PathLike,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    highpass_hz: float = DEFAULT_HIGHPASS_HZ,
    noise_gate: bool = True,
) -> str:
    """Convierte a PCM 16 kHz/mono/16-bit + paso alto SIN compresión loudnorm.

    Esta es la referencia de energía para
    :func:`audio_vad.filter_segments_by_relative_energy`: loudnorm (EBU R128)
    de la Fase 1 comprime el rango dinámico y 'sube' la voz lejana del
    examinador, enmascarando la fuga de auriculares. Esta variante aplica solo
    el paso alto a 80 Hz y la puerta de ruido (opcional), preservando la
    energía relativa de cada tramo de voz.

    Returns:
        La ruta absoluta de ``output_path`` generado.
    """
    binary = _find_ffmpeg()
    absolute_output = os.path.abspath(output_path)
    out_dir = os.path.dirname(absolute_output)
    if not os.path.isdir(out_dir):
        raise AudioPreprocessingError(f"El directorio de salida no existe: {out_dir}")

    proc = subprocess.run(
        _convert_command(
            binary, input_path, absolute_output, highpass_hz, sample_rate, channels,
            noise_gate=noise_gate,
        ),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AudioPreprocessingError(
            f"FFmpeg falló al convertir '{input_path}':\n{proc.stderr[-2000:]}"
        )
    if not os.path.isfile(absolute_output):
        raise AudioPreprocessingError(
            f"FFmpeg terminó pero no generó la salida esperada: {absolute_output}"
        )
    return absolute_output