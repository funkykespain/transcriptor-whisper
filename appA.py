import streamlit as st
import os
import io
import base64
import json
import re
import random
from pydub import AudioSegment, silence
from openai import OpenAI
from dotenv import load_dotenv

# ================= CONFIGURACIÓN INICIAL =================
load_dotenv()

# Variables de Configuración
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL")
MODEL_NAME = os.getenv("OPENROUTER_MODEL")
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD")

# Configuración por defecto (ahora controlable desde la UI)
DEFAULT_MIN_SILENCE_LEN = 2000 # Bajamos un poco para no perder frases cortas, luego unimos
KEEP_SILENCE = 200             # Margen pequeño para no meter ruido extra

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

# ================= CLIENTE Y HERRAMIENTAS =================

def get_ai_client():
    if not API_KEY: return None
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)

def normalizar_audio(audio: AudioSegment) -> AudioSegment:
    """
    Limpieza técnica básica.
    1. Filtro Pasa-Altos (200Hz): Elimina el 'rumble' grave y ruidos de micrófono.
       Ayuda a que el 'bleed' de los auriculares (que suele ser agudo) se aísle mejor
       o que ruidos de mesa no activen el corte.
    2. Mono y 16kHz.
    """
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    # Filtro pasa altos simple para quitar ruidos graves de fondo
    audio = audio.high_pass_filter(200) 
    return audio

def audio_to_base64(audio_segment: AudioSegment) -> str:
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3", bitrate="32k")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def formatear_tiempo(ms):
    seconds = int(ms / 1000)
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

# ================= LÓGICA DE IA =================

def crear_collage_audio(audio_total: AudioSegment, chunks_ranges: list) -> AudioSegment:
    collage = AudioSegment.empty()
    if not chunks_ranges:
        return audio_total[:60000]

    # Tomamos muestras estratégicas
    num_muestras = min(len(chunks_ranges), 6)
    step = len(chunks_ranges) // num_muestras if num_muestras > 0 else 1
    
    for i in range(0, len(chunks_ranges), step):
        start, end = chunks_ranges[i]
        duracion = end - start
        
        # Cogemos el centro de la intervención para evitar cortes o ruidos iniciales
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
    
    prompt_sistema = """
    Eres un lingüista experto. Escucha el audio. 
    Contiene ESPAÑOL y OTRA lengua extranjera.
    Tu misión es identificar esa OTRA lengua.
    Responde ÚNICAMENTE con el Código ISO 639-1 (2 letras).
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Código ISO:"},
                        {"type": "image_url", "image_url": {"url": f"data:audio/mp3;base64,{b64_audio}"}}
                    ]
                }
            ],
            temperature=0,
            max_tokens=10
        )
        
        raw_text = response.choices[0].message.content.strip().upper()
        patron_idiomas = r'\b(' + '|'.join(MAPA_ISO_IDIOMAS.keys()) + r')\b'
        match = re.search(patron_idiomas, raw_text)
        
        if match:
            iso_code = match.group(1)
            return MAPA_ISO_IDIOMAS.get(iso_code, iso_code), iso_code
        else:
            return "IDIOMA_B", "XX"
            
    except Exception as e:
        return "DESCONOCIDO", "XX"

def transcribir_segmento_forense(client, segment_audio: AudioSegment, lengua_b_nombre: str, lengua_b_iso: str) -> dict:
    # Normalizamos (Mono/16k/Filtro) antes de enviar
    b64_audio = audio_to_base64(normalizar_audio(segment_audio))
    
    prompt_sistema = f"""
    Eres un PERITO TRANSCRIPTOR FORENSE. 
    Contexto: Examen oral. Idiomas: ESPAÑOL (ES) y {lengua_b_nombre.upper()} ({lengua_b_iso}).
    
    INSTRUCCIONES:
    1. Transcribe LITERALMENTE lo que dice el ALUMNO (voz principal).
    2. IGNORA voces de fondo muy bajas (profesor/auriculares) si no son el hablante principal.
    3. NO corrijas errores.
    4. Formato JSON estricto.
    
    JSON Output:
    {{
        "idioma": "ES" o "{lengua_b_iso}",
        "texto": "..."
    }}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "JSON:"},
                        {"type": "image_url", "image_url": {"url": f"data:audio/mp3;base64,{b64_audio}"}}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        content = json.loads(response.choices[0].message.content)
        if isinstance(content, list):
            return content[0] if content else {"idioma": "??", "texto": ""}
        return content
        
    except Exception as e:
        return {"idioma": "ERROR", "texto": f"[Error: {str(e)}]"}

# ================= INTERFAZ GRÁFICA =================

st.set_page_config(page_title="Transcriptor Bilateral Pro", page_icon="🎙️", layout="wide")

st.title("🎙️ Transcriptor de Exámenes (Modo Forense)")
st.markdown(f"Motor: **{MODEL_NAME}** | Detección inteligente de 'Bleed' de auriculares")

# --- CONTROL DE ACCESO ---
if ACCESS_PASSWORD:
    pwd = st.sidebar.text_input("🔑 Clave Docente", type="password")
    if pwd != ACCESS_PASSWORD:
        st.warning("Introduce la clave para continuar.")
        st.stop()

# --- CONFIGURACIÓN LATERAL ---
with st.sidebar:
    st.header("🎚️ Calibración de Audio")
    st.info("Ajusta esto para ignorar el sonido de los auriculares.")
    
    # SLIDER CRÍTICO: Umbral relativo al pico máximo
    umbral_dB = st.slider(
        "Sensibilidad del Silencio (dB)", 
        min_value=-50, 
        max_value=-10, 
        value=-28, 
        help="Valor más alto (ej: -20) ignora más ruido de fondo. Valor más bajo (ej: -40) detecta susurros."
    )
    
    min_silence = st.number_input("Silencio Mínimo (ms)", value=2500, step=500, help="Pausas menores a esto se consideran parte de la misma frase.")

client = get_ai_client()
if not client:
    st.error("Falta API KEY")
    st.stop()

uploaded_file = st.file_uploader("📂 Subir Audio (MP3, AAC, M4A, WAV)", type=['mp3', 'm4a', 'wav', 'aac'])

if uploaded_file and st.button("▶️ PROCESAR EXAMEN", type="primary"):
    
    with st.status("Analizando audio...", expanded=True) as status:
        
        # 1. CARGA
        st.write("📥 Cargando y analizando niveles de volumen...")
        audio_total = AudioSegment.from_file(uploaded_file)
        
        # Análisis de niveles para debugging visual
        max_peak = audio_total.max_dBFS
        avg_dB = audio_total.dBFS
        silence_thresh_calc = max_peak + umbral_dB
        
        # Mostramos métricas para que entiendas qué está pasando
        col_met1, col_met2, col_met3 = st.columns(3)
        col_met1.metric("Volumen Pico", f"{max_peak:.2f} dB")
        col_met2.metric("Umbral de Corte", f"{silence_thresh_calc:.2f} dB")
        col_met3.metric("Promedio Audio", f"{avg_dB:.2f} dB")
        
        if silence_thresh_calc > avg_dB:
            st.warning("⚠️ CUIDADO: El umbral de corte es más alto que el promedio del audio. Es posible que cortes voz.")

        # 2. SEGMENTACIÓN (Intento 1: Configuración del Usuario)
        st.write("✂️ Segmentando por voz principal (Pasada 1)...")
        chunks_ranges = silence.detect_nonsilent(
            audio_total,
            min_silence_len=min_silence,
            silence_thresh=silence_thresh_calc,
            seek_step=100
        )
        
        # --- LÓGICA DE RESCATE (NUEVO) ---
        if not chunks_ranges:
            st.warning(f"⚠️ La configuración estricta ({umbral_dB}dB / {min_silence}ms) no detectó voz. Activando MODO RESCATE...")
            
            # Configuración de rescate: Más permisiva (-40dB del pico) y pausas más cortas (1s)
            rescue_thresh = max_peak - 45
            chunks_ranges = silence.detect_nonsilent(
                audio_total,
                min_silence_len=1000,
                silence_thresh=rescue_thresh,
                seek_step=100
            )
            
            if chunks_ranges:
                st.success(f"✅ ¡Rescate exitoso! Se han recuperado {len(chunks_ranges)} intervenciones bajando la sensibilidad.")
            else:
                st.error("❌ El audio parece estar vacío o el volumen es extremadamente bajo incluso para el modo rescate.")
                st.stop()
        else:
            st.write(f"✅ Intervenciones detectadas: {len(chunks_ranges)}")
        
        # 3. DETECCIÓN IDIOMA
        st.write("🌍 Identificando Lengua B...")
        collage = crear_collage_audio(audio_total, chunks_ranges)
        nombre_lb, iso_lb = detectar_lengua_b(client, collage)
        st.success(f"Idioma B: {nombre_lb} ({iso_lb})")
        
        # 4. TRANSCRIPCIÓN
        output_buffer = io.StringIO()
        output_buffer.write(f"EXAMEN: {uploaded_file.name}\n")
        output_buffer.write(f"IDIOMAS: ES - {iso_lb}\n")
        output_buffer.write("="*40 + "\n\n")
        
        progress_bar = st.progress(0)
        
        for i, (start, end) in enumerate(chunks_ranges):
            # Margen de seguridad
            start_adj = max(0, start - KEEP_SILENCE)
            end_adj = min(len(audio_total), end + KEEP_SILENCE)
            segmento = audio_total[start_adj:end_adj]
            
            # Transcripción
            datos = transcribir_segmento_forense(client, segmento, nombre_lb, iso_lb)
            
            ts = formatear_tiempo(start)
            idioma = datos.get("idioma", "??")
            texto = datos.get("texto", "")
            
            bloque = f"[{ts}] [{idioma}]\n{texto}\n\n"
            output_buffer.write(bloque)
            progress_bar.progress((i + 1) / len(chunks_ranges))
            
        status.update(label="Proceso Terminado", state="complete", expanded=False)

    # --- RESULTADOS ---
    st.divider()
    
    # Layout Pro: Reproductor + Texto
    col_izq, col_der = st.columns([1, 2])
    
    with col_izq:
        st.markdown("### 🎧 Audio")
        uploaded_file.seek(0)
        st.audio(uploaded_file)
        
        st.markdown("### 📥 Descarga")
        texto_final = output_buffer.getvalue()
        st.download_button(
            "Descargar TXT", 
            data=texto_final, 
            file_name=f"Acta_{iso_lb}.txt",
            mime="text/plain",
            type="primary"
        )
        
    with col_der:
        st.markdown("### 📜 Transcripción")
        st.text_area("Resultado:", value=texto_final, height=600)