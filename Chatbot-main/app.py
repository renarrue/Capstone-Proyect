# Versión 25.1 (CORREGIDA: Rutas absolutas para Deploy)
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from supabase import create_client, Client
import streamlit_authenticator as stauth
import time
from datetime import time as dt_time
import bcrypt
import pandas as pd

# --- URLs DE LOGOS ---
LOGO_BANNER_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Logo_DuocUC.svg/2560px-Logo_DuocUC.svg.png"
LOGO_ICON_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlve2kMlU53cq9Tl0DMxP0Ffo0JNap2dXq4q_uSdf4PyFZ9uraw7MU5irI6mA-HG8byNI&usqp=CAU"

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Chatbot Duoc UC",
    page_icon=LOGO_ICON_URL,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CARGAR CSS DESDE ARCHIVO EXTERNO (CORREGIDO CON RUTA ABSOLUTA) ---
def load_css(file_name):
    try:
        # Obtener la ruta del directorio donde está este script (app.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Combinar directorio con el nombre del archivo
        css_path = os.path.join(current_dir, file_name)
        
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"⚠️ No se encontró el archivo {file_name}. Asegúrate de que esté en la misma carpeta que app.py.")

# Cargar estilos visuales
load_css("styles.css")

# --- DICCIONARIO DE TRADUCCIONES (ACTUALIZADO CON CHIPS) ---
TEXTS = {
    "es": {
        "label": "Español 🇨🇱",
        "title": "Asistente Académico Duoc UC",
        "sidebar_lang": "Idioma / Language",
        "login_success": "Usuario:",
        "logout_btn": "Cerrar Sesión",
        "tab1": "💬 Chatbot Reglamento",
        "tab2": "📅 Inscripción de Asignaturas",
        "tab3": "🔐 Admin / Auditoría",
        "login_title": "Iniciar Sesión",
        "login_user": "Correo Institucional",
        "login_pass": "Contraseña",
        "login_btn": "Ingresar",
        "login_failed": "❌ Credenciales inválidas",
        "login_welcome": "¡Bienvenido al Asistente!",
        "chat_clear_btn": "🧹 Limpiar Conversación",
        "chat_cleaning": "Procesando solicitud...",
        "chat_cleaned": "¡Historial limpiado!",
        "chat_welcome": "¡Hola **{name}**! 👋 Soy tu asistente virtual de Duoc UC. Pregúntame sobre el reglamento, asistencia o notas.",
        "chat_welcome_clean": "¡Hola **{name}**! El historial ha sido archivado. ¿En qué más te ayudo?",
        "chat_placeholder": "Ej: ¿Con qué nota apruebo el ramo?",
        "chat_thinking": "Consultando reglamento...",
        "feedback_thanks": "¡Gracias por tu feedback! 👍",
        "feedback_report_sent": "Reporte enviado.",
        "feedback_modal_title": "¿Qué podemos mejorar?",
        "feedback_modal_placeholder": "Ej: La información sobre asistencia no es precisa...",
        "btn_send": "Enviar Comentario",
        "btn_cancel": "Omitir",
        "enroll_title": "Toma de Ramos 2025",
        "filter_career": "📂 Filtrar por Carrera:",
        "filter_sem": "⏳ Filtrar por Semestre:",
        "filter_all": "Todas las Carreras",
        "filter_all_m": "Todos los Semestres",
        "reset_btn": "🔄 Limpiar Filtros",
        "search_label": "📚 Buscar Asignatura:",
        "search_placeholder": "Escribe el nombre del ramo...",
        "sec_title": "Secciones Disponibles para:",
        "btn_enroll": "Inscribir",
        "btn_full": "Sin Cupos",
        "msg_enrolled": "✅ ¡Asignatura inscrita exitosamente!",
        "msg_conflict": "⛔ Error: Tope de Horario detectado",
        "msg_already": "ℹ️ Ya estás inscrito en esta asignatura.",
        "my_schedule": "Tu Carga Académica",
        "no_schedule": "No tienes ramos inscritos.",
        "btn_drop": "Anular Ramo",
        "msg_dropped": "Asignatura eliminada de tu carga.",
        "admin_title": "Panel de Control (Admin)",
        "admin_pass_label": "Clave de Acceso:",
        "admin_success": "Acceso Autorizado",
        "admin_info": "Registro de interacciones y feedback negativo.",
        "admin_update_btn": "🔄 Refrescar Datos",
        "col_date": "Fecha/Hora",
        "col_status": "Estado",
        "col_q": "Pregunta Estudiante",
        "col_a": "Respuesta IA",
        "col_val": "Eval",
        "col_com": "Detalle",
        "reg_header": "Crear Cuenta Alumno",
        "reg_name": "Nombre y Apellido",
        "reg_email": "Correo Duoc",
        "reg_pass": "Crear Contraseña",
        "reg_btn": "Registrarse",
        "reg_success": "¡Cuenta creada! Accede desde el Login.",
        "auth_error": "Verifica tus datos.",
        # --- NUEVAS TRADUCCIONES PARA CHIPS ---
        "sug_header": "💡 **¿No sabes qué preguntar? Prueba con esto:**",
        "sug_btn1": "📋 Justificar Inasistencia",
        "sug_query1": "¿Cómo justifico una inasistencia?",
        "sug_btn2": "🎓 Requisitos Titulación",
        "sug_query2": "¿Cuáles son los requisitos para titularme?",
        "
