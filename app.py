import streamlit as st
import os
import io
import base64
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment, silence
from openai import OpenAI
from dotenv import load_dotenv

# ================= CONFIGURACIÓN INICIAL =================
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL")
MODEL_NAME = os.getenv("OPENROUTER_MODEL")
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD")

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

def normalizar_audio(audio: AudioSegment) -> AudioSegment:
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
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

def generar_onda_visual(audio_segment):
    """Genera una imagen simple de la onda de audio para referencia visual."""
    # Convertimos a array de numpy
    samples = np.array(audio_segment.get_array_of_samples())
    
    # Si es estéreo (aunque normalizamos antes), tomamos un canal
    if audio_segment.channels == 2:
        samples = samples[::2]
        
    # Submuestreo para que el gráfico no pese demasiado (1 de cada 100 muestras)
    samples = samples[::100]

    fig, ax = plt.subplots(figsize=(10, 1.5)) # Ancho y bajito
    ax.plot(samples, color='#1E88E5', alpha=0.6, linewidth=0.5)
    ax.axis('off') # Quitamos ejes y bordes
    fig.patch.set_alpha(0) # Fondo transparente
    
    # Convertimos plot a imagen para streamlit
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
        
        # Fórmula: Umbral = Promedio - 10dB (Margen seguridad)
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
                                {"type": "image_url", "image_url": {"url": f"data:audio/mp3;base64,{b64_audio}"}}]
                }
            ],
            temperature=0, max_tokens=10
        )
        raw_text = response.choices[0].message.content.strip().upper()
        patron_idiomas = r'\b(' + '|'.join(MAPA_ISO_IDIOMAS.keys()) + r')\b'
        match = re.search(patron_idiomas, raw_text)
        if match:
            iso_code = match.group(1)
            return MAPA_ISO_IDIOMAS.get(iso_code, iso_code), iso_code
        else: return "IDIOMA_B", "XX"
    except: return "DESCONOCIDO", "XX"

def transcribir_segmento_forense(client, segment_audio: AudioSegment, lengua_b_nombre: str, lengua_b_iso: str) -> dict:
    b64_audio = audio_to_base64(normalizar_audio(segment_audio))
    prompt_sistema = f"""
    Eres un PERITO TRANSCRIPTOR FORENSE. 
    Contexto: Examen oral. Idiomas: ESPAÑOL (ES) y {lengua_b_nombre.upper()} ({lengua_b_iso}).
    INSTRUCCIONES:
    1. Transcribe LITERALMENTE lo que dice el ALUMNO.
    2. Si solo hay ruido, silencio o respiración, devuelve texto vacío.
    3. NO escribas la palabra 'JSON' ni repitas instrucciones.
    4. Formato JSON estricto.
    Output: {{"idioma": "ES" o "{lengua_b_iso}", "texto": "..."}}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": "Analiza y transcribe:"},
                                {"type": "image_url", "image_url": {"url": f"data:audio/mp3;base64,{b64_audio}"}}]
                }
            ],
            response_format={"type": "json_object"}, temperature=0
        )
        content = json.loads(response.choices[0].message.content)
        resultado = content[0] if isinstance(content, list) and content else content
        
        texto_limpio = resultado.get("texto", "").strip()
        if texto_limpio in ["JSON", "json", "JSON:", "undefined"] or not texto_limpio:
            return {"idioma": "??", "texto": ""}

        return resultado
    except Exception as e: return {"idioma": "ERROR", "texto": f"[Error: {str(e)}]"}

# ================= INTERFAZ GRÁFICA (UI/UX ACADÉMICA) =================

st.set_page_config(page_title="Transcriptor Bilateral", page_icon="🎓", layout="wide")

# --- GESTIÓN DE ESTADO ---
if 'umbral_db' not in st.session_state: st.session_state['umbral_db'] = -28
if 'min_silence_ms' not in st.session_state: st.session_state['min_silence_ms'] = 2000
if 'file_id' not in st.session_state: st.session_state['file_id'] = None
if 'calibrado' not in st.session_state: st.session_state['calibrado'] = False
if 'waveform_img' not in st.session_state: st.session_state['waveform_img'] = None

# --- CABECERA PRINCIPAL ---
st.title("🎓 Transcripción de Exámenes")
st.markdown("""
**Asignatura: Interpretación Bilateral** Esta herramienta automatiza la creación del acta de examen.  
1. **Sube el archivo de audio** del alumno.
2. El sistema **detecta automáticamente** el segundo idioma.
3. Se genera una **transcripción literal** (sin correcciones gramaticales) para su evaluación.
""")
st.divider()

# --- LOGIN ---
if ACCESS_PASSWORD:
    pwd = st.sidebar.text_input("🔑 Clave Docente", type="password")
    if pwd != ACCESS_PASSWORD:
        st.warning("Introduce la clave para acceder a la herramienta.")
        st.stop()

# --- SIDEBAR LIMPIO ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    mostrar_ajustes = st.checkbox("Ajustes manuales para ajuste fino", value=False)
    
    if mostrar_ajustes:
        st.info("Solo modifica esto si la transcripción corta palabras o incluye ruido.")
        
        # Slider de dB
        umbral_db_slider = st.slider(
            "Sensibilidad (dB)", -60, -10, st.session_state['umbral_db'], key='slider_db',
            help="Define qué volumen se considera 'Silencio'.\n- Más a la izquierda (-60): Detecta susurros (cuidado con el ruido).\n- Más a la derecha (-10): Ignora ruidos (cuidado con cortar voz)."
        )
        
        # Input de Segundos (Convertimos a ms para el código)
        silencio_sec_default = st.session_state['min_silence_ms'] / 1000
        min_silence_sec = st.number_input(
            "Silencio Mínimo (s)", 
            min_value=0.5, max_value=5.0, 
            value=silencio_sec_default, step=0.5,
            help="Tiempo mínimo de pausa para considerar que ha terminado una frase.\n- Recomendado: 2.0 segundos."
        )
        
        # Guardamos conversión
        st.session_state['umbral_db'] = umbral_db_slider
        st.session_state['min_silence_ms'] = int(min_silence_sec * 1000)
    else:
        st.success("✅ Configuración Automática Activa")

client = get_ai_client()
if not client: st.error("Error: API KEY no configurada"); st.stop()

# --- ZONA DE CARGA ---
uploaded_file = st.file_uploader("📂 Selecciona el archivo de audio (MP3, M4A, AAC, WAV)", type=['mp3', 'm4a', 'wav', 'aac'])

if uploaded_file:
    # Auto-calibración al cambiar archivo
    file_id_actual = uploaded_file.name + str(uploaded_file.size)
    if st.session_state['file_id'] != file_id_actual:
        with st.spinner("🔄 Analizando calidad del audio..."):
            nuevo_umbral, _, _ = autocalibrar_audio(uploaded_file)
            st.session_state['umbral_db'] = nuevo_umbral
            st.session_state['min_silence_ms'] = 2000
            st.session_state['file_id'] = file_id_actual
            st.session_state['calibrado'] = True
            
            # Generamos la onda visual UNA vez y la guardamos
            uploaded_file.seek(0)
            audio_temp = AudioSegment.from_file(uploaded_file)
            st.session_state['waveform_img'] = generar_onda_visual(audio_temp)
            
            st.rerun()

    # Panel de estado simple
    if st.session_state['calibrado']:
        st.success("✅ Audio listo. Calidad óptima detectada.")

    # Botón de acción
    if st.button("▶️ GENERAR ACTA DE EXAMEN", type="primary"):
        with st.status("Procesando examen...", expanded=True) as status:
            
            uploaded_file.seek(0)
            audio_total = AudioSegment.from_file(uploaded_file)
            max_peak = audio_total.max_dBFS
            thresh = max_peak + st.session_state['umbral_db']
            
            st.write("✂️ Detectando intervenciones del alumno...")
            chunks = silence.detect_nonsilent(audio_total, min_silence_len=st.session_state['min_silence_ms'], silence_thresh=thresh, seek_step=100)
            
            if not chunks: # Rescate
                st.warning("⚠️ Voz muy baja. Reintentando con alta sensibilidad...")
                chunks = silence.detect_nonsilent(audio_total, min_silence_len=1000, silence_thresh=max_peak-50, seek_step=100)
            
            if not chunks: st.error("❌ Audio vacío o irreconocible."); st.stop()
            st.write(f"✅ {len(chunks)} intervenciones localizadas.")
            
            st.write("🌍 Identificando idioma B")
            collage = crear_collage_audio(audio_total, chunks)
            nombre_lb, iso_lb = detectar_lengua_b(client, collage)
            
            st.write("📝 Transcribiendo...")
            out_buf = io.StringIO()
            out_buf.write(f"ALUMNO/EXAMEN: {uploaded_file.name}\n")
            out_buf.write(f"IDIOMAS DETECTADOS: ESPAÑOL (ES) - {nombre_lb} ({iso_lb})\n")
            out_buf.write("-" * 50 + "\n\n")
            
            prog = st.progress(0)
            for i, (start, end) in enumerate(chunks):
                seg = audio_total[max(0, start-200):min(len(audio_total), end+200)]
                dat = transcribir_segmento_forense(client, seg, nombre_lb, iso_lb)
                bloque = f"[{formatear_tiempo(start)}] [{dat.get('idioma','??')}]\n{dat.get('texto','')}\n\n"
                out_buf.write(bloque)
                prog.progress((i+1)/len(chunks))
            
            st.session_state['resultado_texto'] = out_buf.getvalue()
            st.session_state['resultado_nombre'] = f"Acta_{uploaded_file.name}_{iso_lb}.txt"
            status.update(label="¡Proceso Completado!", state="complete", expanded=False)

# --- ZONA DE RESULTADOS (DISEÑO ERGONÓMICO) ---
if 'resultado_texto' in st.session_state:
    st.divider()
    st.subheader("🎧 Revisión y Evaluación")
    
    # 1. ONDA VISUAL (Mapa del examen)
    if st.session_state['waveform_img']:
        st.image(st.session_state['waveform_img'], use_container_width=True)
    
    # 2. REPRODUCTOR (Ancho completo)
    uploaded_file.seek(0)
    st.audio(uploaded_file)
    
    # 3. TEXTO (Centrado y con scroll limitado)
    st.markdown("### 📜 Acta Transcrita")
    st.text_area(
        label="Texto del examen",
        value=st.session_state['resultado_texto'],
        height=400, # Altura fija para que el audio no se pierda al hacer scroll
        label_visibility="collapsed"
    )
    
    # 4. DESCARGA (Debajo del texto)
    st.download_button(
        label="📥 Descargar Acta en TXT",
        data=st.session_state['resultado_texto'],
        file_name=st.session_state['resultado_nombre'],
        mime="text/plain",
        type="primary",
        use_container_width=True
    )