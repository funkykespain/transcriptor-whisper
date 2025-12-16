import streamlit as st
import os
import io
import requests
import time
from collections import Counter
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pydub import AudioSegment, silence

# ================= CONFIGURACIÓN =================
# AHORA ES SEGURO: No hay contraseñas aquí.
# Se leen de las variables de entorno (Easypanel) o secrets de Streamlit.

# Intentamos leer de variables de entorno primero
API_URL = os.getenv("WHISPER_URL")
USUARIO = os.getenv("WHISPER_USER")
CONTRASENA = os.getenv("WHISPER_PASS")

MIN_SILENCE_LEN = 2000 
SILENCE_THRESH_OFFSET = -16 
KEEP_SILENCE = 500
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_CONSECUTIVE_ERRORS = 3

# ================= FUNCIONES =================

def get_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.auth = HTTPBasicAuth(USUARIO, CONTRASENA)
    return session

def verificar_servidor():
    if not API_URL or not USUARIO or not CONTRASENA:
        return False, "⚠️ Faltan credenciales. Configura las Variables de Entorno."
        
    try:
        requests.post(API_URL, auth=HTTPBasicAuth(USUARIO, CONTRASENA), timeout=10)
        return True, "✅ Servidor Online"
    except Exception as e:
        return False, f"❌ Error de conexión: {str(e)}"

def formatear_tiempo(ms):
    seconds = int(ms / 1000)
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def transcribir_chunk(session, chunk_audio, filename_ref, language=None, status_placeholder=None):
    buffer = io.BytesIO()
    # Exportamos a 32k para velocidad
    chunk_audio.export(buffer, format="mp3", bitrate="32k") 
    buffer.seek(0)
    file_bytes = buffer.getvalue()
    
    params = {'task': 'transcribe', 'output': 'json'}
    if language:
        params['language'] = language

    for intento in range(1, MAX_RETRIES + 1):
        try:
            buffer_envio = io.BytesIO(file_bytes)
            files = {'audio_file': (f'{filename_ref}.mp3', buffer_envio, 'audio/mpeg')}
            
            response = session.post(API_URL, files=files, params=params, timeout=300)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code in [502, 503, 504]:
                if status_placeholder:
                    status_placeholder.warning(f"Servidor ocupado (502). Reintentando {intento}/{MAX_RETRIES}...")
                time.sleep(RETRY_DELAY)
                continue
            else:
                raise Exception(f"API Error: {response.status_code}")
                
        except Exception as e:
            if intento == MAX_RETRIES:
                raise e
            time.sleep(RETRY_DELAY)

    raise Exception("Max retries")

# ================= INTERFAZ STREAMLIT =================

st.set_page_config(page_title="Transcriptor Bilateral", page_icon="🎙️")

st.title("🎙️ Transcriptor de Exámenes Bilaterales")
st.markdown("Sube tu archivo de audio. El sistema gestionará el cambio de idioma automáticamente.")

# Sidebar de estado y configuración
st.sidebar.header("Configuración")

# Check de variables de entorno
if not API_URL or not USUARIO or not CONTRASENA:
    st.sidebar.error("❌ Faltan Variables de Entorno")
    st.sidebar.info("""
    Para que la app funcione, debes configurar estas variables en Easypanel:
    - `WHISPER_URL`
    - `WHISPER_USER`
    - `WHISPER_PASS`
    """)
    st.stop()

server_ok, msg = verificar_servidor()
if server_ok:
    st.sidebar.success(msg)
else:
    st.sidebar.error(msg)
    st.stop()

uploaded_file = st.file_uploader("Elige un archivo de audio", type=['mp3', 'm4a', 'wav', 'ogg', 'flac'])

if uploaded_file is not None:
    if st.button("🚀 Iniciar Transcripción"):
        
        # 1. Cargar Audio
        with st.status("Procesando audio...", expanded=True) as status:
            st.write("📥 Leyendo archivo y convirtiendo...")
            try:
                audio = AudioSegment.from_file(uploaded_file)
                st.write(f"✅ Audio cargado. Duración: {len(audio)/1000:.1f}s")
            except Exception as e:
                status.update(label="Error al cargar audio", state="error")
                st.error(f"No se pudo leer el audio: {e}")
                st.stop()

            # 2. Detectar Silencios
            st.write("✂️ Detectando intervenciones...")
            silence_thresh = audio.dBFS + SILENCE_THRESH_OFFSET
            chunks_ranges = silence.detect_nonsilent(
                audio,
                min_silence_len=MIN_SILENCE_LEN,
                silence_thresh=silence_thresh,
                seek_step=100
            )
            
            segmentos = []
            for i, (start, end) in enumerate(chunks_ranges):
                start_adj = max(0, start - KEEP_SILENCE)
                end_adj = min(len(audio), end + KEEP_SILENCE)
                segmentos.append({
                    "id": i,
                    "start": start,
                    "audio": audio[start_adj:end_adj],
                    "text": "",
                    "lang": "",
                    "error": False
                })
            
            st.write(f"✅ Se detectaron {len(segmentos)} fragmentos.")
            status.update(label="Transcribiendo...", state="running")

            # 3. Transcribir (Pasada 1)
            session = get_session()
            progress_bar = st.progress(0)
            status_text = st.empty()
            consecutive_errors = 0
            
            for i, seg in enumerate(segmentos):
                status_text.text(f"Transcribiendo fragmento {i+1}/{len(segmentos)}...")
                try:
                    res = transcribir_chunk(session, seg["audio"], f"chunk_{i}", status_placeholder=st)
                    seg["text"] = res.get("text", "").strip()
                    seg["lang"] = res.get("language", "unknown")
                    consecutive_errors = 0
                except Exception as e:
                    seg["error"] = True
                    seg["text"] = "[Error de conexión]"
                    consecutive_errors += 1
                
                progress_bar.progress((i + 1) / len(segmentos))
                
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    st.error("Se han producido demasiados errores consecutivos. Abortando.")
                    break
            
            # 4. Análisis y Corrección
            status.update(label="Analizando idiomas...", state="running")
            valid_langs = [s["lang"] for s in segmentos if not s["error"] and s["lang"] not in ["unknown", "nn"]]
            segundo_idioma = None
            
            if valid_langs:
                count = Counter(valid_langs)
                idiomas_no_es = [l for l in valid_langs if l != 'es']
                if idiomas_no_es:
                    segundo_idioma = Counter(idiomas_no_es).most_common(1)[0][0]
                    st.write(f"🎯 Segundo idioma detectado: **{segundo_idioma.upper()}**")
            
            # 5. Pasada 2 (Corrección)
            if segundo_idioma:
                corregir = [s for s in segmentos if not s["error"] and s["lang"] != 'es' and s["lang"] != segundo_idioma]
                if corregir:
                    st.write(f"🛠 Corrigiendo {len(corregir)} fragmentos...")
                    prog_corr = st.progress(0)
                    for j, seg in enumerate(corregir):
                        try:
                            res = transcribir_chunk(session, seg["audio"], f"fix_{seg['id']}", language=segundo_idioma)
                            seg["text"] = res.get("text", "").strip()
                            seg["lang"] = segundo_idioma
                        except:
                            pass
                        prog_corr.progress((j+1)/len(corregir))
            
            status.update(label="¡Completado!", state="complete", expanded=False)

        # 6. Generar Resultado
        output_io = io.StringIO()
        output_io.write(f"Archivo: {uploaded_file.name}\n")
        output_io.write(f"Segundo idioma: {segundo_idioma or 'N/A'}\n")
        output_io.write("="*50 + "\n\n")
        
        for seg in segmentos:
            t = formatear_tiempo(seg["start"])
            if seg["error"]:
                output_io.write(f"{t} - ERROR - Fallo de conexión\n\n")
            else:
                output_io.write(f"{t} - {seg['lang']} - {seg['text']}\n\n")
        
        st.success("Transcripción finalizada.")
        st.download_button(
            label="💾 Descargar Transcripción (.txt)",
            data=output_io.getvalue(),
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcrito.txt",
            mime="text/plain"
        )
