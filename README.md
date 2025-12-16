# 🎙️ Transcriptor Bilateral (Whisper + Streamlit)

Herramienta web diseñada para automatizar la transcripción de exámenes de interpretación bilateral. Esta aplicación no procesa el audio en local, sino que actúa como un cliente inteligente que se conecta a un servidor privado de Whisper (Docker).

## 🚀 Características

- **Detección de Intervenciones:** Corta el audio basándose en los silencios (pausas) entre hablantes.
- **Gestión Bilingüe Inteligente:**
  1. Transcribe cada fragmento detectando el idioma automáticamente.
  2. Analiza estadísticamente los idiomas predominantes.
  3. Realiza una segunda pasada para corregir fragmentos con idiomas mal detectados.
- **Optimización de Red:** Comprime los fragmentos de audio en vuelo (MP3 32k) para evitar timeouts en conexiones lentas o servidores saturados.
- **Interfaz Gráfica:** Subida de archivos, barras de progreso y descarga directa del TXT final.

## 🛠️ Requisitos de Despliegue

Este proyecto está diseñado para desplegarse fácilmente en **Easypanel, Coolify** o cualquier entorno compatible con Docker.

### Variables de Entorno (OBLIGATORIAS)

Para que la aplicación funcione, debes configurar las siguientes variables en tu panel de hosting:

| Variable       | Descripción                                      | Ejemplo                          |
|----------------|--------------------------------------------------|----------------------------------|
| `WHISPER_URL`  | URL de tu backend Whisper (endpoint completo)    | `https://whisper.midominio.com/asr` |
| `WHISPER_USER` | Usuario para Basic Auth                          | `admin`                          |
| `WHISPER_PASS` | Contraseña para Basic Auth                       | `mi_contraseña_segura`           |

## 🐳 Despliegue en Easypanel

1. Crea un nuevo servicio de tipo **App**.
2. En **Source**, conecta este repositorio de GitHub.
3. En **Build**, asegúrate de que use el **Dockerfile** incluido en la raíz.
4. En **Environment**, añade las 3 variables mencionadas arriba.
5. En **Network / Domains**:
   - Container Port: `8501`
   - Asigna tu dominio público.
6. ¡Desplegar! 🚀

## 💻 Desarrollo Local

Si quieres ejecutarlo en tu ordenador:

1. **Instala FFmpeg** (Requisito del sistema).
2. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
