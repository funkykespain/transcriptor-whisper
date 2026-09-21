#!/usr/bin/env python
"""Diagnóstico de un tramo concreto del pipeline (sin modificar lógica de producción).

Uso (desde la raíz del repo):
    venv/bin/python scripts/diagnostico_segmento.py \
        --archivo "/home/kyke/Descargas/Transcript/Bilateral/bilateral italiano elena garcia claro (1).aac" \
        --segundo 574 [--iso-b auto|it] [--asr local|remoto|ambos] [--contexto asr|exacto]

Reproduce los pasos EXACTOS de app.py (importando las funciones de producción bajo
"import app") y, para el tramo solicitado + los 2 adyacentes a cada lado, imprime:

  1. Audio & VAD        -> timestamps + duración exacta del chunk
  2. LID de audio       -> forced_language + confianza + top probabilidades
  3. ASR bruto          -> texto de Whisper ANTES de la llamada del LLM
                          (faster-whisper local, language=None; opcional el
                          endpoint remoto self-hosted usado por la versión antigua)
  4. Respuesta LLM      -> texto + campo "idioma" del JSON (transcribir_segmento_forense)
  5. NLP de texto       -> detect_text_language (lingua/langdetect, solo es|B)
                          sobre el ASR bruto y sobre el texto del LLM, con su score
  6. historial_contexto -> contenido exacto pasado al prompt del tramo
  7. Etiqueta final     -> idioma_detectado + regla híbrida que la activó
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Asegurar que los módulos locales del repo son importables.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

logging.getLogger("streamlit").setLevel(logging.CRITICAL)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# ---------------- Lógica de PRODUCCIÓN (solo importada, no modificada) --------
import app as production  # noqa: E402
import audio_lid as lid  # noqa: E402
from audio_lid import (  # noqa: E402
    DEFAULT_LANGUAGE,
    detect_language_for_segment,
    detect_text_language,
    load_audio_array,
)
from audio_preprocessing import convert_base_pcm, preprocess_audio_base  # noqa: E402
from audio_vad import (  # noqa: E402
    DEFAULT_MIN_SPEECH_DURATION_MS,
    DEFAULT_THRESHOLD,
    compute_segments_energy_db,
    detect_speech_segments,
    filter_segments_by_relative_energy,
    speech_segments_to_ranges,
)

ARCHIVO_POR_DEFECTO = ("/home/kyke/Descargas/Transcript/Bilateral/"
                       "bilateral italiano elena garcia claro (1).aac")
SEGUNDO_POR_DEFECTO = 574
AMPLITUD = 2          # segmentos de contexto a cada lado del objetivo
MIN_SILENCE_MS = 2000  # valor por defecto de la sesión en app.py

# Endpoint self-hosted usado por la versión antigua (solo diagnóstico opcional).
REMOTO_ASR_URL = "https://atu-whisper.bp1xn4.easypanel.host/asr"
REMOTO_ASR_USER = "bilateral"
REMOTO_ASR_PASS = "C0ntr45egn4"


def _nlp_score(texto, iso_a, iso_b):
    """(lang, confianza) del backend NLP restringido a es|B (solo diagnóstico)."""
    if not texto or not texto.strip():
        return None, None
    detector = lid._text_detector_for([iso_a, iso_b])
    if detector is None:
        return None, None
    try:
        return detector(texto[:2000])
    except Exception:  # noqa: BLE001
        return None, None


def _asr_local(modelo, audio_array, start_sample, end_sample, initial_prompt=None):
    """ASR bruto local (faster-whisper, language=None -> literal, sin traducir)."""
    chunk = audio_array[start_sample:end_sample]
    segmentos, _info = modelo.transcribe(
        chunk, language=None, beam_size=1, vad_filter=False,
        initial_prompt=initial_prompt,
    )
    return " ".join(s.text for s in segmentos).strip()


def _asr_remoto(audio_pydub):
    """ASR bruto remoto (endpoint self-hosted de la versión antigua)."""
    try:
        import requests
    except ImportError:
        return None, "requests no instalado"
    buffer = io.BytesIO()
    audio_pydub.export(buffer, format="mp3", bitrate="32k")
    buffer.seek(0)
    try:
        resp = requests.post(
            REMOTO_ASR_URL,
            auth=(REMOTO_ASR_USER, REMOTO_ASR_PASS),
            files={"audio_file": ("chunk.mp3", buffer.getvalue(), "audio/mpeg")},
            params={"task": "transcribe", "output": "json"},
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"error de conexión: {exc}"
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"
    data = resp.json()
    return data.get("text", "").strip(), f"lang_auto={data.get('language')}"


def _imprimir_tramo(foco, info):
    print("\n" + "=" * 78)
    print(f"TRAMO [{info['idx']}]  {info['etiqueta_hora']}  "
          f"({foco.start:.3f} s -> {foco.end:.3f} s  duración={foco.end - foco.start:.3f} s)")
    print("=" * 78)

    print("\n[1] AUDIO & VAD")
    print(f"    start={foco.start:.3f} s | end={foco.end:.3f} s | "
          f"duración={foco.end - foco.start:.3f} s")
    print(f"    energía RMS (referencia SIN loudnorm): {info['energia_dbfs']:.1f} dBFS")
    micro = (foco.end - foco.start) < 1.5
    print(f"    regla duración: {'MICRO-CHUNK <1.5s (sin LID Whisper)' if micro else 'LID normal'}")

    print("\n[2] LID DE AUDIO (audio_lid)")
    print(f"    forced_language={info['forced_language']!r}")
    print(f"    idioma_objetivo (Paso B -> directiva de transcripción): {info['idioma_objetivo']!r}")
    print(f"    confianza={info['confianza_lid']:.3f}  probabilidades_top="
          f"{json.dumps(info['probs_top'], ensure_ascii=False)}")

    print("\n[3] ASR BRUTO (antes de la llamada al LLM)")
    print(f"    local SIN initial_prompt (language=None): {info.get('asr_local_limpio') or info['asr_local']!r}")
    print(f"    local CON  initial_prompt bilingüe      : {info['asr_local']!r}")
    if info.get("asr_remoto"):
        print(f"    remoto (atu-whisper): {info['asr_remoto']!r} [{info['asr_remoto_meta']}]")

    print("\n[4] RESPUESTA DEL LLM (transcribir_segmento_forense)")
    print(f"    JSON: {json.dumps(info['llm_response'], ensure_ascii=False)}")
    print(f"    texto_llm={info['texto_llm']!r}")
    print(f"    idioma(JSON LLM)={info['idioma_llm']!r}   <- se ignora para la etiqueta")

    print("\n[5] NLP DE TEXTO (lingua/langdetect, restringido [es, B])")
    for nombre, texto in (("ASR bruto", info["asr_local"]),
                          ("texto LLM", info["texto_llm"])):
        lang, conf = _nlp_score(texto, "es", info["iso_lb"])
        print(f"    sobre {nombre:11s}: lang={lang!r} "
              f"conf={None if conf is None else round(conf, 3)}")
    print(f"    detect_text_language(texto_llm, es, {info['iso_lb']}) -> "
          f"{info['texto_lengua_b']!r}")

    print("\n[6] ESTADO DEL CONTEXTO (historial_contexto pasado al prompt)")
    print(f"    idioma_previo={info['idioma_previo']!r}")
    print(f"    historial_contexto ({len(info['historial'])} chars) =")
    print("      " + (info["historial"] or "'' (vacío)").replace("\n", "\\n"))

    print("\n[7] ETIQUETA FINAL")
    print(f"    idioma_detectado={info['idioma_final']!r}")
    print(f"    REGLA: {info['regla']}")


def main():
    parser = argparse.ArgumentParser(description="Diagnóstico de un tramo del pipeline.")
    parser.add_argument("--archivo", default=ARCHIVO_POR_DEFECTO)
    parser.add_argument("--segundo", type=float, default=SEGUNDO_POR_DEFECTO)
    parser.add_argument("--iso-b", default="auto", help="auto (LLM) o código ISO, p. ej. it")
    parser.add_argument("--asr", choices=["local", "remoto", "ambos"], default="local")
    parser.add_argument("--contexto", choices=["asr", "exacto"], default="asr",
                        help="asr: contexto previo con ASR local (rápido); "
                             "exacto: reproduce TODAS las llamadas LLM anteriores")
    args = parser.parse_args()

    archivo = args.archivo
    if not os.path.isfile(archivo):
        print(f"ERROR: archivo no encontrado: {archivo}", file=sys.stderr)
        sys.exit(1)
    print(f"ARCHIVO: {archivo}")
    print(f"SEGUNDO A INSPECCIONAR: {args.segundo} s "
          f"({production.formatear_tiempo(int(args.segundo * 1000))})")

    with tempfile.TemporaryDirectory(prefix="diag_lid_") as tmp:
        ext = Path(archivo).suffix or ""
        src = os.path.join(tmp, "entrada" + ext)
        shutil.copyfile(archivo, src)
        norm = os.path.join(tmp, "norm.wav")
        energy = os.path.join(tmp, "energy.wav")

        print("\n— Fase 1: preprocesado (16k/mono/16-bit, -18 LUFS) + referencia de energía …")
        preprocess_audio_base(src, norm)
        convert_base_pcm(src, energy)

        print("— Fase 2: VAD (Silero) + filtro de energía relativa …")
        segments_vad = detect_speech_segments(
            norm,
            threshold=DEFAULT_THRESHOLD,
            min_speech_duration_ms=DEFAULT_MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=MIN_SILENCE_MS,
        )
        segments_vad = filter_segments_by_relative_energy(energy, segments_vad)
        ranges = speech_segments_to_ranges(segments_vad)
        if not segments_vad:
            print("ERROR: el VAD no localizó voz.", file=sys.stderr)
            sys.exit(1)
        energias = compute_segments_energy_db(energy, segments_vad)
        print(f"→ {len(segments_vad)} tramos detectados.")

        # Lengua B (producción: LLM; o fijada por CLI)
        client = production.get_ai_client()
        iso_lb = "it"
        if args.iso_b.lower() != "auto":
            iso_lb = args.iso_b.lower()
        elif client is not None:
            print("— Fase 4 (Lengua B) con LLM de producción …")
            from pydub import AudioSegment as _AS

            audio_total = _AS.from_file(archivo)
            collage = production.crear_collage_audio(audio_total, ranges)
            _nombre, iso_lb = production.detectar_lengua_b(client, collage)
        else:
            print("  (!) Sin cliente LLM; Lengua B = 'it' (por el nombre del archivo).")
        allowed_languages = ["es", iso_lb.lower()]
        print(f"→ Lengua B = {iso_lb.upper()} | allowed_languages={allowed_languages}")

        objetivo = min(
            range(len(segments_vad)),
            key=lambda i: abs((segments_vad[i].start + segments_vad[i].end) / 2 - args.segundo),
        )
        foco_indices = sorted({k for k in range(objetivo - AMPLITUD, objetivo + AMPLITUD + 1)
                               if 0 <= k < len(segments_vad)})

        lid_audio = load_audio_array(norm)
        modelo_asr = lid.get_lid_model()

        # --- Réplica del bucle de producción (acumula contexto en orden) ---
        historial = ""
        idioma_anterior_lid = DEFAULT_LANGUAGE
        fin_prev = None
        idioma_previo = "ES"
        recorte = {}

        for i, seg in enumerate(segments_vad):
            start_ms, end_ms = ranges[i]
            es_foco = i in foco_indices
            # Estado entrante (lo que se pasa al prompt/LLM de este tramo)
            historial_entrada = historial
            idioma_previo_entrada = idioma_previo

            forced_language, det = detect_language_for_segment(
                lid_audio, seg, default=DEFAULT_LANGUAGE,
                previous_language=idioma_anterior_lid,
                previous_end=fin_prev,
                allowed_languages=allowed_languages,
            )
            fin_prev = seg.end

            beam_start = int(round(seg.start * 16000))
            beam_end = int(round(seg.end * 16000))
            prompt_asr = (
                f"Transcripción bilingüe en castellano y {iso_lb.upper()} ({iso_lb}). "
                "Transcribir literalmente las palabras pronunciadas sin traducir ni normalizar."
            )
            asr_local = _asr_local(modelo_asr, lid_audio, beam_start, beam_end,
                                   initial_prompt=prompt_asr)
            # Variante sin initial_prompt para comparar el efecto en la ventana.
            asr_local_limpio = (_asr_local(modelo_asr, lid_audio, beam_start, beam_end)
                                if es_foco else asr_local)
            from pydub import AudioSegment as _AS2

            asr_remoto = asr_remoto_meta = None
            if es_foco and args.asr in ("remoto", "ambos"):
                audio_total_seg = _AS2.from_file(archivo)
                seg_audio = audio_total_seg[start_ms:min(end_ms, len(audio_total_seg))]
                asr_remoto, asr_remoto_meta = _asr_remoto(seg_audio)

            if es_foco:
                audio_total_seg = _AS2.from_file(archivo)
                seg_audio = audio_total_seg[start_ms:min(end_ms, len(audio_total_seg))]
                llm_response = production.transcribir_segmento_forense(
                    client, seg_audio, iso_lb.upper(), iso_lb, historial, idioma_previo,
                    idioma_objetivo=forced_language,
                )
            else:
                # Fuera de la ventana: texto == ASR bruto (aproximación al LLM).
                llm_response = {"idioma": forced_language.upper(), "texto": asr_local}

            texto_segmento = (llm_response.get("texto") or "").strip()
            if texto_segmento:
                historial += f" {texto_segmento}"
                if len(historial) > 800:
                    historial = historial[-800:]

            if es_foco:
                texto_lengua_b = detect_text_language(texto_segmento, "es", iso_lb)
                if texto_lengua_b == iso_lb.lower():
                    idioma_final = iso_lb.upper()
                    regla = f"híbrido: NLP texto -> Lengua B ({iso_lb.upper()})"
                else:
                    idioma_final = forced_language.upper()
                    regla = "híbrido: prima LID de audio (sin marca de Lengua B en el texto)"
                if (seg.end - seg.start) < 1.5:
                    regla += " | tramo <1.5s: el LID audio no se ejecutó (heredado)"
            else:
                idioma_final = forced_language.upper()
                regla = "(fuera de la ventana de inspección)"

            idioma_anterior_lid = idioma_final.lower()
            idioma_previo = idioma_final

            if es_foco:
                recorte[i] = {
                    "idx": i,
                    "etiqueta_hora": production.formatear_tiempo(start_ms),
                    "energia_dbfs": energias[i],
                    "forced_language": forced_language,
                    "idioma_objetivo": forced_language.lower(),
                    "confianza_lid": det.confidence,
                    "probs_top": dict(list(det.probabilities.items())[:5]),
                    "asr_local": asr_local,
                    "asr_local_limpio": asr_local_limpio,
                    "asr_remoto": asr_remoto,
                    "asr_remoto_meta": asr_remoto_meta or "",
                    "llm_response": llm_response,
                    "texto_llm": texto_segmento,
                    "idioma_llm": llm_response.get("idioma"),
                    "texto_lengua_b": texto_lengua_b,
                    "historial": historial_entrada,
                    "idioma_previo": idioma_previo_entrada,
                    "idioma_final": idioma_final,
                    "regla": regla,
                    "iso_lb": iso_lb,
                }

        print(f"\n→ Tramo objetivo (centrado en ~{args.segundo}s): índice {objetivo} "
              f"de {len(segments_vad)} | ventana inspeccionada: {foco_indices}")
        for i in foco_indices:
            _imprimir_tramo(segments_vad[i], recorte[i])

        print("\n" + "=" * 78)
        print("DIAGNÓSTICO COMPLETO.")
        print("(historial_contexto de tramos previos construido con ASR local 'language=None';")
        print(" usa --contexto exacto para repetir TODAS las llamadas LLM anteriores.")


if __name__ == "__main__":
    main()