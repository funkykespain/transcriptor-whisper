import streamlit as st
import logging
import os
import io
import base64
import json
import re
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment, silence
from openai import OpenAI
from dotenv import load_dotenv

# Módulos locales (audio_preprocessing, audio_vad): asegura su import aunque
# la app se lance desde otro directorio o entornos tipo AppTest/testing.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_preprocessing import preprocess_audio_base
from audio_lid import (
    DEFAULT_LANGUAGE,
    detect_language_for_segment,
    load_audio_array,
)
from audio_vad import (
    detect_speech_segments,
    export_speech_chunks,
    speech_segments_to_ranges,
)

logger = logging.getLogger(__name__)

# ================= CONFIGURACIÓN INICIAL =================
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL")
MODEL_NAME = os.getenv("OPENROUTER_MODEL")
KOFI_URL = "https://ko-fi.com/S6S61TZEJ8"
raw_passwords = os.getenv("ACCESS_PASSWORD", "")
VALID_PASSWORDS = [p.strip() for p in raw_passwords.split(",") if p.strip()]

# ================= ESTILOS CSS (Footer y UI) =================
st.markdown("""
<style>
    /* Estilo para el Footer Fijo en la Sidebar */
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f0f2f6; /* Color gris claro estándar de sidebar */
        padding: 15px 20px;
        z-index: 999;
        border-top: 1px solid #dcdcdc;
        font-family: sans-serif;
    }
    
    /* Modo oscuro compatible para el footer */
    @media (prefers-color-scheme: dark) {
        .sidebar-footer {
            background-color: #262730;
            border-top: 1px solid #41424b;
        }
    }

    /* Ajuste para que el contenido del sidebar no quede tapado por el footer */
    section[data-testid="stSidebar"] > div:first-child {
        padding-bottom: 120px;
    }

    /* Estilo del contenedor Ko-fi */
    .kofi-container {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .kofi-text {
        font-size: 0.85em;
        margin-bottom: 10px;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ================= CONFIGURACIÓN DE IDIOMAS =================
MAPA_ISO_IDIOMAS = {
    'HR': 'CROATA', 'HY': 'ARMENIO', 'KO': 'COREANO', 'EN': 'INGLÉS',
    'FR': 'FRANCÉS', 'IT': 'ITALIANO', 'DE': 'ALEMÁN', 'PT': 'PORTUGUÉS',
    'NL': 'NEERLANDÉS', 'SV': 'SUECO', 'DA': 'DANÉS', 'FI': 'FINLANDÉS',
    'NO': 'NORUEGO', 'IS': 'ISLANDÉS', 'RU': 'RUSO', 'PL': 'POLACO',
    'RO': 'RUMANO', 'CS': 'CHECO', 'SK': 'ESLOVACO', 'HU': 'HÚNGARO',
    'BG': 'BÚLGARO', 'SR': 'SERBIO', 'UK': 'UCRANIANO', 'EL': 'GRIEGO',
    'SL': 'ESLOVENO', 'ET': 'ESTONIO', 'LV': 'LETÓN', 'LT': 'LITUANO',
    'ZH': 'CHINO', 'JA': 'JAPONÉS', 'AR': 'ÁRABE', 'HI': 'HINDI',
    'TR': 'TURCO', 'HE': 'HEBREO', 'VI': 'VIETNAMITA', 'TH': 'TAILANDÉS',
    'ID': 'INDONESIO', 'FA': 'PERSA', 'CA': 'CATALÁN', 'GL': 'GALLEGO',
    'EU': 'EUSKERA'
}

# ================= HERRAMIENTAS Y FUNCIONES =================

def get_ai_client():
    if not API_KEY: return None
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)

def _normalizar_audio_legacy(audio: AudioSegment) -> AudioSegment:
    """Normalización mínima con pydub (fallback si el pipeline FFmpeg no está)."""
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    # Filtro Acústico Equilibrado (agnóstico al micro y la voz):
    # 100 Hz elimina el retumbe grave de fondo sin tijeretear la voz tenue
    # ni los finales de frase cuando el alumno baja la voz.
    audio = audio.high_pass_filter(100)
    return audio

def normalizar_audio(audio: AudioSegment) -> AudioSegment:
    """Pre-tratamiento homogéneo de la entrada (Fase 1 del pipeline ASR).

    Delega en :func:`audio_preprocessing.preprocess_audio_base` para obtener
    PCM 16 kHz / mono / 16-bit, paso alto a 80 Hz y normalización EBU R128 a
    -18 LUFS. Si FFmpeg no está disponible o falla, se conserva el
    comportamiento legacy de pydub para no romper el flujo de la aplicación.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "normalizar_in.wav")
            out_path = os.path.join(tmp, "normalizar_out.wav")
            audio.export(in_path, format="wav")
            preprocess_audio_base(in_path, out_path)
            return AudioSegment.from_file(out_path, format="wav")
    except Exception as exc:  # noqa: BLE001 - fallback controlado de todo el pipeline
        logger.warning("Pipeline FFmpeg no disponible (%s); usamos normalización legacy.", exc)
        return _normalizar_audio_legacy(audio)

def audio_to_base64(audio_segment: AudioSegment) -> str:
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3", bitrate="32k")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def formatear_tiempo(ms):
    seconds = int(ms / 1000)
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def generar_onda_visual(audio_segment):
    """Genera una imagen simple de la onda de audio para referencia visual."""
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples[::2]
    samples = samples[::100]

    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.plot(samples, color='#1E88E5', alpha=0.6, linewidth=0.5)
    ax.axis('off')
    fig.patch.set_alpha(0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf

# ================= LÓGICA DE AUTO-CALIBRACIÓN =================

def autocalibrar_audio(uploaded_file):
    try:
        audio = AudioSegment.from_file(uploaded_file)
        peak = audio.max_dBFS
        avg = audio.dBFS
        
        target_threshold = avg - 10 
        suggested_slider = target_threshold - peak
        suggested_slider = max(-60, min(-10, int(suggested_slider)))
        
        return suggested_slider, peak, avg
    except:
        return -28, 0, 0

# ================= LÓGICA DE IA (DETECTAR Y TRANSCRIBIR) =================

def crear_collage_audio(audio_total: AudioSegment, chunks_ranges: list) -> AudioSegment:
    collage = AudioSegment.empty()
    if not chunks_ranges: return audio_total[:60000]

    num_muestras = min(len(chunks_ranges), 6)
    step = len(chunks_ranges) // num_muestras if num_muestras > 0 else 1
    
    for i in range(0, len(chunks_ranges), step):
        start, end = chunks_ranges[i]
        duracion = end - start
        if duracion > 8000:
            mid = start + (duracion // 2)
            clip = audio_total[mid - 3000 : mid + 3000] 
        else:
            clip = audio_total[start:end]
        collage += clip
        if len(collage) > 50000: break
            
    return normalizar_audio(collage)

def detectar_lengua_b(client, audio_collage: AudioSegment) -> tuple:
    b64_audio = audio_to_base64(audio_collage)
    prompt_sistema = "Eres un lingüista experto. Identifica la LENGUA EXTRANJERA (no Español) en el audio. Responde SOLO con el código ISO 639-1 (2 letras)."
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": "Código ISO:"},
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": b64_audio,
                                        "format": "mp3"
                                    }
                                }]
                }
            ],
            temperature=0,
            top_p=1,
            max_tokens=10
        )
        raw_text = response.choices[0].message.content.strip().upper()
        patron_idiomas = r'\b(' + '|'.join(MAPA_ISO_IDIOMAS.keys()) + r')\b'
        match = re.search(patron_idiomas, raw_text)
        if match:
            iso_code = match.group(1)
            return MAPA_ISO_IDIOMAS.get(iso_code, iso_code), iso_code
        else: return "IDIOMA_B", "XX"
    except: return "DESCONOCIDO", "XX"

def transcribir_segmento_forense(client, segment_audio: AudioSegment, lengua_b_nombre: str, lengua_b_iso: str, contexto_previo: str, idioma_previo: str, forced_language: str = "") -> dict:
    # 1. Normalización
    b64_audio = audio_to_base64(normalizar_audio(segment_audio))
    # Fase 4 (LID): `forced_language` es el ISO detectado por audio_lid
    # ('es' | 'it' | ...) para este fragmento. Con una ASR local (Whisper) se
    # pasaría directamente como language=forced_language; aquí se inyecta al
    # prompt pericial a continuación para impedir alucinaciones/traducciones.
    if forced_language:
        logger.debug("LID forzado del segmento: %s", forced_language)
    
    # 2. Prompt Forense General (Principios Periciales Universales)
    prompt_sistema = f"""
    Eres un PERITO TRANSCRIPTOR FORENSE con especialización en análisis acústico y lingüístico multilingüe.
    Contexto: Examen de Interpretación Bilateral.
    Idiomas involucrados: ESPAÑOL (ES) y {lengua_b_nombre.upper()} ({lengua_b_iso}).

    MEMORIA PREVIA (solo referencia): "...{contexto_previo[-300:]}" - Idioma reportado en el fragmento anterior: {idioma_previo}

    PRINCIPIOS PERICIALES UNIVERSALES:

    a) FIDELIDAD FONÉTICA Y NO CORRECCIÓN:
       - Transcribe de manera estrictamente literal las ondas sonoras del audio.
       - Si el hablante comete una incorrección gramatical, utiliza una variante no estándar, pronuncia mal una palabra o inventa un término, DEBES transcribir la forma exacta percibida en el audio.
       - Queda estrictamente prohibido normalizar, corregir ortográficamente o estandarizar el vocabulario al idioma normativo.

    b) IDENTIFICACIÓN PRECISA DE IDIOMA B:
       - Analiza la estructura y los términos del fragmento de audio ACTUAL para asignar el código de idioma correcto (ES o {lengua_b_iso}).
       - Las frases cortas o con elementos propios de la Lengua B deben etiquetarse adecuadamente, superando la inercia del contexto en español cuando el alumno cambie de idioma.

    c) DISCRIMINACIÓN DE RUIDO MECÁNICO:
       - Si el segmento solo contiene ruidos de fondo (paso de páginas, roces, tos) sin habla humana inteligible, devuelve únicamente el texto vacío "".

    d) AISLAMIENTO DE MEMORIA:
       - Utiliza el contexto previo SOLO como referencia para resolver ambigüedades del fragmento actual.
       - Queda estrictamente prohibido traducir, repetir o incluir en la respuesta texto proveniente de la memoria previa que no corresponda al audio actual.

    FORMATO: Responde únicamente en JSON estricto.

    Output: {{"idioma": "ES" o "{lengua_b_iso}", "texto": "..."}}
    """

    # Fase 4 (LID): fuerza el idioma detectado por audio_lid en ESTE fragmento.
    if forced_language:
        prompt_sistema += (
            f'\n    LID DEL FRAGMENTO: el idioma detectado por LID es "{forced_language.upper()}".\n'
            "    Regla FORZADA: transcribe este fragmento literalmente en ese idioma;\n"
            "    está terminantemente prohibido traducirlo o cambiar de idioma.\n"
        )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": b64_audio,
                                "format": "mp3"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"}, 
            temperature=0,
            top_p=1
        )
        
        # --- VALIDACIONES ---
        if not response or not response.choices: return {"idioma": "ERROR", "texto": ""}
        mensaje = response.choices[0].message
        if not mensaje or not mensaje.content: return {"idioma": "??", "texto": ""}

        try:
            content = json.loads(mensaje.content)
        except json.JSONDecodeError: return {"idioma": "ERROR", "texto": ""}
            
        if isinstance(content, list): resultado = content[0] if content else {}
        else: resultado = content
            
        if not isinstance(resultado, dict): return {"idioma": "??", "texto": ""}
        
        texto_raw = resultado.get("texto", "").strip()
        
        # Único filtro técnico (no lingüístico): rechaza marcadores de placeholder
        # que el modelo puede emitir cuando no produce contenido real.
        if texto_raw.lower() in ["json", "undefined", "null"]:
            return {"idioma": "??", "texto": ""}
        
        # Filtro Técnico Anti-Echo (Python): si la transcripción (mayor a 15 caracteres)
        # reproduce literalmente la cola final de la memoria previa, se descarta como eco.
        # Previene repeticiones de memoria en fragmentos que solo contienen silencio o ruido blanco.
        if len(texto_raw) > 15 and texto_raw in contexto_previo[-300:]:
            return {"idioma": "??", "texto": ""}
        
        # Sin listas ni reglas hardcodeadas: la discriminación de ruido y el
        # aislamiento de memoria los resuelve el modelo pericial mediante los
        # principios universales definidos en el prompt.
        resultado["texto"] = texto_raw
        return resultado

    except Exception as e:
        return {"idioma": "ERROR", "texto": f"[Error: {str(e)}]"}

# ================= UI PRINCIPAL =================

st.set_page_config(page_title="Transcriptor Bilateral", page_icon="🎓", layout="wide")

# --- GESTIÓN DE ESTADO ---
if 'umbral_db' not in st.session_state: st.session_state['umbral_db'] = -28
if 'min_silence_ms' not in st.session_state: st.session_state['min_silence_ms'] = 2000
if 'file_id' not in st.session_state: st.session_state['file_id'] = None
if 'calibrado' not in st.session_state: st.session_state['calibrado'] = False
if 'waveform_img' not in st.session_state: st.session_state['waveform_img'] = None

# --- SIDEBAR + FOOTER FIJO ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Footer GitHub
    st.markdown(
        """
        <div class="sidebar-footer">
            <div style="text-align: center;">
                <a href="https://github.com/funkykespain/transcriptor-whisper" target="_blank" 
                   style="color: inherit; text-decoration: none; font-size: 0.85rem; display: flex; align-items: center; justify-content: center; gap: 8px; opacity: 0.7;">
                   <svg height="20" viewBox="0 0 16 16" version="1.1" width="20" aria-hidden="true" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
                   Repositorio & Docs
                </a>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- CABECERA ---
st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <img src="https://raw.githubusercontent.com/funkykespain/transcriptor-whisper/refs/heads/main/profile.png" 
             alt="Logo" 
             style="width: 70px; height: 70px; border-radius: 10px; object-fit: cover;">
        <h1 style="margin: 0; padding: 0; font-size: 3rem;">Transcripción de Exámenes</h1>
    </div>
""", unsafe_allow_html=True)
st.markdown("""

**Asignatura: Interpretación Bilateral** Esta herramienta automatiza la creación del acta de examen.

1. **Sube el archivo de audio** del alumno.

2. El primer idioma será español (ES). El sistema **detecta automáticamente** el segundo idioma.

3. Se genera una **transcripción literal** (evitando en lo posible correcciones gramaticales) para su evaluación.

""")
st.divider()

# --- LOGIN CON KO-FI ---
# Si la lista de contraseñas no está vacía, activamos el bloqueo
if VALID_PASSWORDS:
    pwd = st.sidebar.text_input("🔑 Clave Docente", type="password")
    
    # Comprobamos si la clave escrita NO está en la lista de válidas
    if pwd not in VALID_PASSWORDS:
        st.warning("🔒 Herramienta protegida. Introduce la Clave Docente en >> Configuración.")
        
        st.markdown(f"""
        <div class="kofi-container">
            <p class="kofi-text">¿No tienes clave? Apoya el proyecto con un café (mínimo 3€) y recibirás tu Clave Docente en tu correo electrónico personal:</p>
            <a href='{KOFI_URL}' target='_blank'>
                <img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi2.png?v=6' border='0' alt='Invítame a un Café en ko-fi.com' />
            </a>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

# --- AJUSTES MANUALES (Solo si hay acceso) ---
with st.sidebar:
    mostrar_ajustes = st.checkbox("Ajustes manuales para ajuste fino", value=False)
    if mostrar_ajustes:
        st.info("Solo modifica esto si la transcripción corta palabras o incluye ruido.")
        umbral_db_slider = st.slider("Sensibilidad (dB)", -60, -10, st.session_state['umbral_db'], key='slider_db', help="Define qué volumen se considera 'Silencio'.\n- Más a la izquierda (-60): Detecta susurros (cuidado con el ruido).\n- Más a la derecha (-10): Ignora ruidos (cuidado con cortar voz).")
        silencio_sec_default = st.session_state['min_silence_ms'] / 1000
        min_silence_sec = st.number_input("Silencio Mínimo (s)", 0.5, 5.0, value=silencio_sec_default, step=0.5, help="Tiempo mínimo de pausa para considerar que ha terminado una frase.\n- Recomendado: 2.0 segundos.")
        st.session_state['umbral_db'] = umbral_db_slider
        st.session_state['min_silence_ms'] = int(min_silence_sec * 1000)
    else:
        st.success("✅ Configuración Automática Activa")

client = get_ai_client()
if not client: st.error("Error: API KEY no configurada"); st.stop()

# --- ZONA DE CARGA ---
uploaded_file = st.file_uploader("📂 Selecciona el archivo de audio (MP3, M4A, WAV, AAC)", type=['mp3', 'm4a', 'wav', 'aac'])

if uploaded_file:
    file_id_actual = uploaded_file.name + str(uploaded_file.size)
    if st.session_state['file_id'] != file_id_actual:
        with st.spinner("🔄 Analizando calidad del audio..."):
            nuevo_umbral, _, _ = autocalibrar_audio(uploaded_file)
            st.session_state['umbral_db'] = nuevo_umbral
            st.session_state['min_silence_ms'] = 2000
            st.session_state['file_id'] = file_id_actual
            st.session_state['calibrado'] = True
            
            uploaded_file.seek(0)
            audio_temp = AudioSegment.from_file(uploaded_file)
            st.session_state['waveform_img'] = generar_onda_visual(audio_temp)
            st.rerun()

    if st.session_state['calibrado']: st.success("✅ Audio listo. Calidad óptima detectada.")

    if st.button("▶️ GENERAR ACTA DE EXAMEN", type="primary"):
        with st.status("Procesando examen...", expanded=True) as status:
            with tempfile.TemporaryDirectory() as tmp_work:
            
                uploaded_file.seek(0)
                audio_total = AudioSegment.from_file(uploaded_file)
                max_peak = audio_total.max_dBFS
                thresh = max_peak + st.session_state['umbral_db']
                
                # --- FASE 1: pre-tratamiento del audio completo (16 kHz / mono / -18 LUFS) ---
                st.write("🎛️ Pre-tratando audio (F1: 16 kHz / mono / EBU R128 -18 LUFS)...")
                st.session_state['chunks_exportados'] = []
                chunks, segments_vad = [], []
                try:
                    src_path = os.path.join(tmp_work, "entrada_audio")
                    with open(src_path, "wb") as fh:
                        fh.write(uploaded_file.getvalue())
                    audio_norm_path = os.path.join(tmp_work, "audio_normalizado.wav")
                    preprocess_audio_base(src_path, audio_norm_path)
                    
                    # --- FASE 2: voz con Silero VAD + padding de 400 ms ---
                    st.write("✂️ Detectando intervenciones con Silero VAD (F2)...")
                    segments_vad = detect_speech_segments(
                        audio_norm_path,
                        threshold=0.5,
                        min_silence_duration_ms=st.session_state['min_silence_ms'],
                    )
                    # Timestamps (ms) relativos al audio original (misma duración)
                    chunks = speech_segments_to_ranges(segments_vad)
                    if segments_vad:
                        chunks_dir = tempfile.mkdtemp(prefix="chunks_vad_")
                        st.session_state['chunks_exportados'] = export_speech_chunks(
                            audio_norm_path, segments_vad, chunks_dir
                        )
                except Exception:
                    logger.warning("Fase 1/2 con fallo; se usa detección clásica.", exc_info=True)
                
                # --- Fallback clásico si el VAD no localiza voz ---
                if not chunks:
                    st.warning("⚠️ VAD sin voz segura. Reintentando con alta sensibilidad...")
                    chunks = silence.detect_nonsilent(audio_total, min_silence_len=st.session_state['min_silence_ms'], silence_thresh=thresh, seek_step=100)
                if not chunks:
                    chunks = silence.detect_nonsilent(audio_total, min_silence_len=1000, silence_thresh=max_peak-50, seek_step=100)
                if not chunks: st.error("❌ Audio vacío o irreconocible."); st.stop()
                st.write(f"✅ {len(chunks)} intervenciones localizadas.")
                
                # Fase 4 (LID): array normalizado 16k para detectar el idioma
                # de cada tramo sobre los primeros segundos del segmento.
                lid_audio = None
                if segments_vad:
                    try:
                        lid_audio = load_audio_array(audio_norm_path)
                    except Exception:
                        logger.warning("LID: no se pudo cargar el audio normalizado.", exc_info=True)
                
                st.write("🌍 Identificando idioma...")
                collage = crear_collage_audio(audio_total, chunks)
                nombre_lb, iso_lb = detectar_lengua_b(client, collage)
                
                st.write("📝 Transcribiendo con contexto inteligente...")
                out_buf = io.StringIO()
                out_buf.write(f"ALUMNO/EXAMEN: {uploaded_file.name}\n")
                out_buf.write(f"IDIOMAS DETECTADOS: ESPAÑOL (ES) - {nombre_lb} ({iso_lb})\n")
                out_buf.write("-" * 50 + "\n\n")
                
                prog = st.progress(0)
                
                # --- BUCLE CON CONTEXTO (Lógica V2.1.0) ---
                historial_contexto = ""
                idioma_actual = "ES"
                
                for i, (start, end) in enumerate(chunks):
                    # Los tramos VAD ya incluyen el padding de 400 ms y son
                    # relativos al audio original: el texto queda sincronizado
                    # con los segundos exactos del examen.
                    seg = audio_total[start:min(end, len(audio_total))]
                    
                    # Fase 4 (LID): idioma forzado para la ASR por segmento.
                    forced_language = DEFAULT_LANGUAGE
                    if lid_audio is not None and segments_vad:
                        forced_language, det = detect_language_for_segment(
                            lid_audio, segments_vad[i], default=DEFAULT_LANGUAGE
                        )
                        if not det.is_confident():
                            logger.debug(
                                "LID con confianza baja (%s); fallback seguro a '%s'.",
                                det.confidence, forced_language,
                            )
                    
                    dat = transcribir_segmento_forense(client, seg, nombre_lb, iso_lb, historial_contexto, idioma_actual, forced_language=forced_language)
                    
                    texto_segmento = dat.get('texto','')
                    idioma_detectado = dat.get('idioma','??')
                    
                    # Actualizar contexto (si hay texto válido)
                    if texto_segmento:
                        historial_contexto += f" {texto_segmento}"
                        if len(historial_contexto) > 800: # Limite para no saturar
                            historial_contexto = historial_contexto[-800:]
                    
                    # Mantener registro del último idioma detectado como contexto de continuidad
                    # (el prompt decide el idioma real del fragmento actual sin sesgo de inercia).
                    if idioma_detectado in ["ES", iso_lb]:
                        idioma_actual = idioma_detectado
                    
                    bloque = f"[{formatear_tiempo(start)}] [{idioma_detectado}]\n{texto_segmento}\n\n"
                    out_buf.write(bloque)
                    prog.progress((i+1)/len(chunks))
                
                st.session_state['resultado_texto'] = out_buf.getvalue()
                st.session_state['resultado_nombre'] = f"Acta_{uploaded_file.name}_{iso_lb}.txt"
                status.update(label="¡Proceso Completado!", state="complete", expanded=False)

# --- RESULTADOS ---
if 'resultado_texto' in st.session_state:
    st.divider()
    st.subheader("🎧 Revisión y Evaluación")
    
    if st.session_state['waveform_img']:
        st.image(st.session_state['waveform_img'], use_container_width=True)
    
    uploaded_file.seek(0)
    st.audio(uploaded_file)
    
    st.markdown("### 📜 Acta Transcrita")
    st.text_area(label="Texto del examen", value=st.session_state['resultado_texto'], height=400, label_visibility="collapsed")
    
    st.download_button(label="📥 Descargar Acta en TXT", data=st.session_state['resultado_texto'], file_name=st.session_state['resultado_nombre'], mime="text/plain", type="primary", use_container_width=True)