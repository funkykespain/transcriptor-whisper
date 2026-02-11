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

def limpiar_repeticiones(texto):
    """
    Detecta y ELIMINA bucles de alucinación (ej: 'la la la la la').
    Diferencia entre un tartamudeo natural (2-3 veces) y un error de IA (+4 veces).
    """
    if not texto: return ""
    
    # 1. Caso extremo: "la la la la la la" (Alucinación de ruido)
    # Si una palabra corta (<=3 letras) se repite más de 4 veces, es ruido casi seguro. Borramos todo.
    patron_ruido = r'\b(\w{1,3})(\s+\1){4,}'
    if re.search(patron_ruido, texto, flags=re.IGNORECASE):
        return "" # Devolvemos vacío, asumimos que era ruido de papel/tos

    # 2. Caso leve: Tartamudeo real o bucle pequeño
    # Si se repite 3 veces, lo dejamos como tartamudeo (ej: "pero pero pero...")
    return texto

def transcribir_segmento_forense(client, segment_audio: AudioSegment, lengua_b_nombre: str, lengua_b_iso: str, contexto_previo: str, idioma_previo: str) -> dict:
    # 1. Normalización
    b64_audio = audio_to_base64(normalizar_audio(segment_audio))
    
    # 2. Prompt Forense Anti-Ruido (Actualizado)
    prompt_sistema = f"""
    Eres un PERITO TRANSCRIPTOR FORENSE. 
    Contexto: Examen de Interpretación Bilateral.
    Idiomas: ESPAÑOL (ES) y {lengua_b_nombre.upper()} ({lengua_b_iso}).
    
    CONTEXTO PREVIO: "...{contexto_previo[-300:]}" (Idioma: {idioma_previo})

    INSTRUCCIONES CLAVE:
    1. TRANSCRIPCIÓN LITERAL (VERBATIM): Escribe EXACTAMENTE lo que escuchas.
    2. PROHIBIDO CORREGIR: NO arregles la gramática, NO mejores el estilo, NO corrijas la pronunciación. Si el alumno dice "yo sabo", escribe "yo sabo".
    3. INERCIA DE IDIOMA: Si el audio es ambiguo, corto o una continuación clara, MANTÉN el idioma anterior ({idioma_previo}). Solo cambia si es evidente.
    4. GESTIÓN DE RUIDO:
         - Si escuchas RUIDO DE PAPEL, GOLPES, TOS o RESPIRACIÓN FUERTE -> NO lo transcribas como "la la la" o sílabas sueltas. Devuelve texto vacío "".
         - Solo transcribe si hay PALABRAS INTELIGIBLES. Si solo hay ruido, devuelve "".
    5. PROHIBIDO REPETIR CONTEXTO: La información de "MEMORIA DE CONTEXTO" es lo que YA se dijo. NO lo vuelvas a escribir. Si el audio actual solo contiene silencio o repite lo anterior, devuelve "".
    6. FORMATO: JSON estricto.
    
    Output: {{"idioma": "ES" o "{lengua_b_iso}", "texto": "..."}}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Transcribe (Ignora ruidos de fondo/papel):"},
                        {"type": "image_url", "image_url": {"url": f"data:audio/mp3;base64,{b64_audio}"}}
                    ]
                }
            ],
            response_format={"type": "json_object"}, 
            temperature=0
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
        
        # Filtros de Alucinación
        if texto_raw.lower() in ["json", "undefined", "null"]:
            return {"idioma": "??", "texto": ""}
        
        # Filtro Anti-Eco (Python):
        # Si el texto transcrito está contenido DENTRO del contexto previo (es una repetición exacta), lo borramos.
        # Usamos los últimos 50 caracteres para comparar.
        if len(texto_raw) > 10 and texto_raw in contexto_previo[-len(texto_raw)-20:]:
             return {"idioma": "??", "texto": ""} # Es un eco, lo borramos
            
        # APLICAMOS EL FILTRO DE REPETICIÓN
        texto_final = limpiar_repeticiones(texto_raw)
        
        resultado["texto"] = texto_final
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
            
            uploaded_file.seek(0)
            audio_total = AudioSegment.from_file(uploaded_file)
            max_peak = audio_total.max_dBFS
            thresh = max_peak + st.session_state['umbral_db']
            
            st.write("✂️ Detectando intervenciones del alumno...")
            chunks = silence.detect_nonsilent(audio_total, min_silence_len=st.session_state['min_silence_ms'], silence_thresh=thresh, seek_step=100)
            if not chunks: 
                st.warning("⚠️ Voz muy baja. Reintentando con alta sensibilidad...")
                chunks = silence.detect_nonsilent(audio_total, min_silence_len=1000, silence_thresh=max_peak-50, seek_step=100)
            if not chunks: st.error("❌ Audio vacío o irreconocible."); st.stop()
            st.write(f"✅ {len(chunks)} intervenciones localizadas.")
            
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
                seg = audio_total[max(0, start-200):min(len(audio_total), end+200)]
                
                # Llamada a la función forense V2.1.0
                dat = transcribir_segmento_forense(client, seg, nombre_lb, iso_lb, historial_contexto, idioma_actual)
                
                texto_segmento = dat.get('texto','')
                idioma_detectado = dat.get('idioma','??')
                
                # Actualizar contexto (si hay texto válido)
                if texto_segmento:
                    historial_contexto += f" {texto_segmento}"
                    if len(historial_contexto) > 800: # Limite para no saturar
                        historial_contexto = historial_contexto[-800:]
                
                # Actualizar inercia de idioma
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