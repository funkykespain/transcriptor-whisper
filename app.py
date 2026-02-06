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
from dotenv import load_dotenv
load_dotenv()

# ================= CONFIGURACIÓN =================
# Lectura de variables de entorno
API_URL = os.getenv("WHISPER_URL")
USUARIO = os.getenv("WHISPER_USER")
CONTRASENA = os.getenv("WHISPER_PASS")
# Variable para proteger el frontend
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD")

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
        return False, "⚠️ Faltan credenciales del backend."
    try:
        requests.post(API_URL, auth=HTTPBasicAuth(USUARIO, CONTRASENA), timeout=10)
        return True, "✅ Servidor de Transcripción Online"
    except Exception as e:
        return False, f"❌ Error de conexión con Whisper: {str(e)}"

def formatear_tiempo(ms):
    seconds = int(ms / 1000)
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def transcribir_chunk(session, chunk_audio, filename_ref, language=None, status_placeholder=None):
    buffer = io.BytesIO()
    # Exportamos a 32k para optimizar la subida a la red interna/externa
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
                    status_placeholder.warning(f"El servidor está procesando una carga alta. Reintentando {intento}/{MAX_RETRIES}...")
                time.sleep(RETRY_DELAY)
                continue
            else:
                raise Exception(f"API Error: {response.status_code}")
                
        except Exception as e:
            if intento == MAX_RETRIES:
                raise e
            time.sleep(RETRY_DELAY)

    raise Exception("Max retries")

# ================= INTERFAZ GRÁFICA (STREAMLIT) =================

st.set_page_config(page_title="Herramienta de Transcripción - Interpretación Bilateral", page_icon="🎓")

st.title("🎓 Transcripción de Exámenes")
st.subheader("Asignatura: Interpretación Bilateral")

st.markdown("""
Esta herramienta automatizada permite generar la transcripción de un examen oral.
El sistema procesará el audio para:
1.  **Detectar intervenciones:** Separar automáticamente los turnos de palabra basándose en los silencios.
2.  **Identificar el idioma:** Distinguir entre Español y la Lengua B (Inglés, Francés, Alemán, Italiano, etc.).
3.  **Generar acta:** Crear un archivo de texto con los códigos de tiempo exactos (MM:SS).
""")

st.divider()

# --- VERIFICACIÓN DE SEGURIDAD (ACCESO PROFESOR) ---
acceso_concedido = False

if ACCESS_PASSWORD:
    col1, col2 = st.columns([2, 3])
    with col1:
        password_input = st.text_input("🔑 Clave de Acceso Docente", type="password", help="Introduce la contraseña para habilitar la transcripción.")
    
    if password_input == ACCESS_PASSWORD:
        st.success("Acceso autorizado")
        acceso_concedido = True
    elif password_input:
        st.error("Clave incorrecta")
else:
    # Si no hay variable de entorno configurada, se permite el paso (modo abierto)
    st.warning("⚠️ Modo sin protección (Variable ACCESS_PASSWORD no configurada en el servidor)")
    acceso_concedido = True

# --- SIDEBAR DE ESTADO ---
st.sidebar.header("Estado del Sistema")
server_ok, msg = verificar_servidor()
if server_ok:
    st.sidebar.success(msg)
else:
    st.sidebar.error(msg)
    st.stop() # Detiene la app si no hay conexión con el backend

# --- ÁREA DE TRABAJO (Solo si hay acceso) ---
if acceso_concedido:
    uploaded_file = st.file_uploader("Seleccione el archivo de audio del examen (MP3, M4A, OGG, WAV)", type=['mp3', 'm4a', 'wav', 'ogg', 'flac'])

    if uploaded_file is not None:
        # Botón principal
        if st.button("🚀 Iniciar Procesamiento del Examen", type="primary"):
            
            # 1. Cargar Audio
            with st.status("Iniciando sistema...", expanded=True) as status:
                st.markdown("**ℹ️ Nota:** Para **CANCELAR** el proceso en cualquier momento, pulse el botón **Stop** (🛑) en la esquina superior derecha o recargue la página.")
                
                st.write("📥 Leyendo metadatos y convirtiendo formato...")
                try:
                    audio = AudioSegment.from_file(uploaded_file)
                    duracion_fmt = formatear_tiempo(len(audio))
                    st.write(f"✅ Audio cargado correctamente. Duración total: **{duracion_fmt}**")
                except Exception as e:
                    status.update(label="Error en el formato de audio", state="error")
                    st.error(f"El archivo está dañado o el formato no es compatible: {e}")
                    st.stop()

                # 2. Detectar Silencios
                st.write("✂️ Segmentando intervenciones por pausas...")
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
                
                st.write(f"✅ Se han detectado **{len(segmentos)} intervenciones** distintas.")
                status.update(label="Transcribiendo intervenciones...", state="running")

                # 3. Transcribir (Pasada 1)
                session = get_session()
                progress_bar = st.progress(0)
                status_text = st.empty()
                consecutive_errors = 0
                
                for i, seg in enumerate(segmentos):
                    status_text.caption(f"Procesando intervención {i+1} de {len(segmentos)}...")
                    try:
                        res = transcribir_chunk(session, seg["audio"], f"chunk_{i}", status_placeholder=st)
                        seg["text"] = res.get("text", "").strip()
                        seg["lang"] = res.get("language", "unknown")
                        consecutive_errors = 0
                    except Exception as e:
                        seg["error"] = True
                        seg["text"] = "[Error de conexión con el servidor]"
                        consecutive_errors += 1
                    
                    progress_bar.progress((i + 1) / len(segmentos))
                    
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        st.error("⛔ Se ha perdido la conexión con el servidor de transcripción. Proceso abortado.")
                        break
                
                # 4. Análisis y Corrección
                status.update(label="Verificando idiomas detectados...", state="running")
                valid_langs = [s["lang"] for s in segmentos if not s["error"] and s["lang"] not in ["unknown", "nn"]]
                segundo_idioma = None
                
                if valid_langs:
                    idiomas_no_es = [l for l in valid_langs if l != 'es']
                    if idiomas_no_es:
                        segundo_idioma = Counter(idiomas_no_es).most_common(1)[0][0]
                        st.markdown(f"🎯 Lengua B detectada: **{segundo_idioma.upper()}**")
                
                # 5. Pasada 2 (Corrección)
                if segundo_idioma:
                    corregir = [s for s in segmentos if not s["error"] and s["lang"] != 'es' and s["lang"] != segundo_idioma]
                    if corregir:
                        st.write(f"🛠 Refinando {len(corregir)} intervenciones...")
                        prog_corr = st.progress(0)
                        for j, seg in enumerate(corregir):
                            try:
                                res = transcribir_chunk(session, seg["audio"], f"fix_{seg['id']}", language=segundo_idioma)
                                seg["text"] = res.get("text", "").strip()
                                seg["lang"] = segundo_idioma
                            except:
                                pass
                            prog_corr.progress((j+1)/len(corregir))
                
                status.update(label="¡Proceso finalizado con éxito!", state="complete", expanded=False)

            # 6. Generar Resultado
            output_io = io.StringIO()
            output_io.write(f"Examen: {uploaded_file.name}\n")
            output_io.write(f"Lengua B detectada: {segundo_idioma.upper() if segundo_idioma else 'No determinada'}\n")
            output_io.write("="*60 + "\n\n")
            
            for seg in segmentos:
                t = formatear_tiempo(seg["start"])
                idioma_display = "🇪🇸 ES" if seg['lang'] == 'es' else f"🌐 {seg['lang'].upper()}"
                
                if seg["error"]:
                    output_io.write(f"[{t}] - ERROR DE SISTEMA\n\n")
                else:
                    output_io.write(f"[{t}] - {idioma_display}\n{seg['text']}\n\n")
            
            st.success("El documento está listo para su descarga.")
            st.download_button(
                label="📥 Descargar Acta de Transcripción (.txt)",
                data=output_io.getvalue(),
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcrito.txt",
                mime="text/plain"
            )
else:
    st.info("🔒 Introduce la clave de acceso docente para desbloquear la herramienta.")
