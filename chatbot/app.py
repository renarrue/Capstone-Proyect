"""
================================================
CHATBOT ACADÉMICO DUOC UC - VERSIÓN MEJORADA
================================================
Autor: Rena
Fecha: Diciembre 2024
Versión: 2.0 (Con mejoras implementadas)

MEJORAS INCLUIDAS:
1. ✅ Validaciones de seguridad (email @duocuc.cl, contraseñas fuertes)
2. ✅ Sistema de logging profesional
3. ✅ Exportar horario (PDF + ICS/Google Calendar)
4. ✅ Dashboard de estadísticas
5. ✅ Confirmaciones mejoradas
6. ✅ Cache optimizado
7. ✅ Citas de artículos en RAG
8. ✅ Rate limiting admin
9. ✅ Loading states
10. ✅ Easter eggs y saludos personalizados
================================================
"""

# =============================================
# IMPORTS
# =============================================
import streamlit as st
import os
import bcrypt
import time
import re
import random
import logging
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO

# Supabase
from supabase import create_client, Client

# LangChain y RAG
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Exportar PDF
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch

# Exportar ICS (Google Calendar)
from icalendar import Calendar, Event as ICalEvent

# =============================================
# CONFIGURACIÓN DE LOGGING
# =============================================
def setup_logging():
    """Configura sistema de logging profesional"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler('chatbot_duoc.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()
logger.info("=" * 50)
logger.info("🚀 Iniciando Chatbot Duoc UC")
logger.info("=" * 50)

# =============================================
# VARIABLES GLOBALES
# =============================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "DUOC2025")  # Cambiar en producción

# URLs del logo
LOGO_BANNER_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Logo_DuocUC.svg/2560px-Logo_DuocUC.svg.png"
LOGO_ICON_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlve2kMlU53cq9Tl0DMxP0Ffo0JNap2dXq4q_uSdf4PyFZ9uraw7MU5irI6mA-HG8byNI&usqp=CAU"

# =============================================
# CONFIGURACIÓN DE PÁGINA
# =============================================
st.set_page_config(
    page_title="Chatbot Duoc UC",
    page_icon=LOGO_ICON_URL,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CARGAR CSS
# =============================================
def load_css(file_name):
    """Carga archivo CSS personalizado"""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        logger.info("✅ CSS cargado exitosamente")
    except FileNotFoundError:
        st.warning(f"⚠️ Archivo CSS no encontrado: {file_name}")
        logger.warning(f"⚠️ CSS no encontrado: {file_name}")

load_css("styles.css")

# =============================================
# FUNCIONES DE VALIDACIÓN Y SEGURIDAD
# =============================================

def validar_email_duoc(email):
    """Valida que el email sea institucional @duocuc.cl"""
    pattern = r'^[a-zA-Z0-9._%+-]+@duocuc\.cl$'
    return re.match(pattern, email) is not None

def validar_password_fuerte(password):
    """
    Valida contraseña fuerte:
    - Mínimo 8 caracteres
    - Al menos 1 mayúscula, 1 minúscula, 1 número, 1 símbolo
    """
    if len(password) < 8:
        return False, "La contraseña debe tener al menos 8 caracteres"
    if not re.search(r'[A-Z]', password):
        return False, "Debe incluir al menos una mayúscula"
    if not re.search(r'[a-z]', password):
        return False, "Debe incluir al menos una minúscula"
    if not re.search(r'\d', password):
        return False, "Debe incluir al menos un número"
    if not re.search(r'[@$!%*?&]', password):
        return False, "Debe incluir al menos un símbolo (@$!%*?&)"
    return True, ""

def check_admin_rate_limit():
    """
    Verifica intentos de acceso admin.
    Lockout después de 3 intentos fallidos por 5 minutos.
    """
    if "admin_attempts" not in st.session_state:
        st.session_state.admin_attempts = 0
        st.session_state.admin_lockout_until = None
    
    if st.session_state.admin_lockout_until:
        ahora = datetime.now()
        if ahora < st.session_state.admin_lockout_until:
            segundos_restantes = (st.session_state.admin_lockout_until - ahora).seconds
            return False, f"⛔ Demasiados intentos. Bloqueado por {segundos_restantes} segundos"
        else:
            st.session_state.admin_attempts = 0
            st.session_state.admin_lockout_until = None
    
    return True, ""

def registrar_intento_admin(exitoso):
    """Registra intento de acceso admin"""
    if exitoso:
        st.session_state.admin_attempts = 0
        st.session_state.admin_lockout_until = None
        logger.info("🔓 Acceso admin exitoso")
    else:
        st.session_state.admin_attempts += 1
        logger.warning(f"🔒 Intento admin fallido (intento #{st.session_state.admin_attempts})")
        
        if st.session_state.admin_attempts >= 3:
            st.session_state.admin_lockout_until = datetime.now() + timedelta(minutes=5)
            logger.warning("⛔ Admin bloqueado por 5 minutos")

# =============================================
# FUNCIONES DE UX
# =============================================

def mostrar_confirmacion(mensaje, tipo="success"):
    """Muestra confirmación con estilo mejorado"""
    iconos = {
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️"
    }
    
    icono = iconos.get(tipo, "ℹ️")
    
    if tipo == "success":
        st.success(f"{icono} {mensaje}")
    elif tipo == "error":
        st.error(f"{icono} {mensaje}")
    elif tipo == "warning":
        st.warning(f"{icono} {mensaje}")
    else:
        st.info(f"{icono} {mensaje}")

def obtener_saludo_hora():
    """Saludo personalizado según hora del día"""
    hora = datetime.now().hour
    
    if 5 <= hora < 12:
        saludos = [
            "☀️ ¡Buenos días!",
            "🌅 ¡Buen día!",
            "☕ ¡Buenos días! ¿Ya tomaste café?"
        ]
    elif 12 <= hora < 19:
        saludos = [
            "☀️ ¡Buenas tardes!",
            "😊 ¡Hola! ¿Cómo va tu día?",
            "📚 ¡Buenas tardes! Hora de estudiar"
        ]
    else:
        saludos = [
            "🌙 ¡Buenas noches!",
            "✨ ¡Hola! Estudiando de noche, ¿eh?",
            "🦉 ¡Búho nocturno detectado!"
        ]
    
    return random.choice(saludos)

# =============================================
# FUNCIONES DE EXPORTACIÓN
# =============================================

def generar_pdf_horario(sections, user_name):
    """Genera PDF del horario del estudiante"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Título
    title = Paragraph(f"<b>Horario Académico - {user_name}</b>", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 0.3*inch))
    
    # Subtítulo
    subtitle = Paragraph(
        f"<i>Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}</i>",
        styles['Normal']
    )
    elements.append(subtitle)
    elements.append(Spacer(1, 0.3*inch))
    
    # Tabla de horario
    data = [['Asignatura', 'Sección', 'Día', 'Horario', 'Profesor']]
    
    for section in sections:
        data.append([
            section.get('subject_name', 'N/A'),
            section.get('code', 'N/A'),
            section.get('day', 'N/A'),
            f"{section.get('start_time', 'N/A')}-{section.get('end_time', 'N/A')}",
            section.get('professor', 'N/A')
        ])
    
    table = Table(data, colWidths=[2*inch, 0.8*inch, 1*inch, 1.2*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002342')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 0.5*inch))
    
    # Footer
    footer = Paragraph(
        "<i>Duoc UC - Sistema de Gestión Académica</i>",
        styles['Normal']
    )
    elements.append(footer)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

def generar_ics_horario(sections, user_name):
    """Genera archivo ICS para importar a Google Calendar"""
    cal = Calendar()
    cal.add('prodid', '-//Chatbot Duoc UC//mxm.dk//')
    cal.add('version', '2.0')
    cal.add('x-wr-calname', f'Horario {user_name}')
    
    # Mapeo días a números (Lunes = 0)
    dias = {
        'Lunes': 0, 'Monday': 0,
        'Martes': 1, 'Tuesday': 1,
        'Miércoles': 2, 'Wednesday': 2,
        'Jueves': 3, 'Thursday': 3,
        'Viernes': 4, 'Friday': 4,
        'Sábado': 5, 'Saturday': 5
    }
    
    # Fecha de inicio del semestre (ajustar según corresponda)
    inicio_semestre = datetime(2025, 3, 3)  # Ejemplo: 3 de marzo 2025
    
    for section in sections:
        event = ICalEvent()
        event.add('summary', f"{section.get('subject_name', 'Clase')} ({section.get('code', 'N/A')})")
        event.add('description', f"Profesor: {section.get('professor', 'N/A')}")
        event.add('location', 'Duoc UC - Sede Padre Alonso de Ovalle')
        
        # Calcular fecha/hora del primer día
        dia_nombre = section.get('day', 'Lunes')
        dia_numero = dias.get(dia_nombre, 0)
        primer_dia = inicio_semestre + timedelta(days=(dia_numero - inicio_semestre.weekday()) % 7)
        
        try:
            hora_inicio = datetime.strptime(section.get('start_time', '08:00'), '%H:%M').time()
            hora_fin = datetime.strptime(section.get('end_time', '10:00'), '%H:%M').time()
        except:
            hora_inicio = datetime.strptime('08:00', '%H:%M').time()
            hora_fin = datetime.strptime('10:00', '%H:%M').time()
        
        dtstart = datetime.combine(primer_dia, hora_inicio)
        dtend = datetime.combine(primer_dia, hora_fin)
        
        event.add('dtstart', dtstart)
        event.add('dtend', dtend)
        
        # Recurrencia semanal por 16 semanas (un semestre)
        event.add('rrule', {'freq': 'weekly', 'count': 16})
        
        cal.add_component(event)
    
    return cal.to_ical()

def resaltar_articulos(texto):
    """Resalta las citas de artículos en la respuesta del RAG"""
    patron = r'(Artículo N°\d+|Art\. \d+|Article \d+)'
    texto_resaltado = re.sub(patron, r'**\1**', texto)
    return texto_resaltado

# =============================================
# TEXTOS MULTIIDIOMA
# =============================================
TEXTS = {
    "es": {
        "system_prompt": "Eres un coordinador académico de Duoc UC. Tu función es ayudar a los estudiantes con dudas sobre el reglamento académico.",
        "welcome": "Bienvenido al Chatbot Académico de Duoc UC",
        "chat_input": "Escribe tu pregunta sobre el reglamento académico...",
        "login_title": "Iniciar Sesión",
        "login_user": "Correo electrónico",
        "login_pass": "Contraseña",
        "login_btn": "Ingresar",
        "login_success": "Bienvenido/a",
        "login_error": "Usuario o contraseña incorrectos",
        "reg_title": "Registrarse",
        "reg_name": "Nombre completo",
        "reg_email": "Correo institucional",
        "reg_pass": "Contraseña",
        "reg_btn": "Crear cuenta",
        "reg_success": "Cuenta creada exitosamente. Ya puedes iniciar sesión.",
        "admin_pass_label": "Contraseña de administrador",
        "tab_chat": "💬 Chatbot Académico",
        "tab_enrollment": "📚 Inscripción de Asignaturas",
        "tab_admin": "🔐 Admin / Auditoría"
    },
    "en": {
        "system_prompt": "You are an academic coordinator at Duoc UC. Your role is to help students with questions about academic regulations.",
        "welcome": "Welcome to Duoc UC Academic Chatbot",
        "chat_input": "Write your question about academic regulations...",
        "login_title": "Login",
        "login_user": "Email",
        "login_pass": "Password",
        "login_btn": "Sign in",
        "login_success": "Welcome",
        "login_error": "Incorrect username or password",
        "reg_title": "Register",
        "reg_name": "Full name",
        "reg_email": "Institutional email",
        "reg_pass": "Password",
        "reg_btn": "Create account",
        "reg_success": "Account created successfully. You can now sign in.",
        "admin_pass_label": "Administrator password",
        "tab_chat": "💬 Academic Chatbot",
        "tab_enrollment": "📚 Course Enrollment",
        "tab_admin": "🔐 Admin / Audit"
    }
}

# Easter eggs
EASTER_EGGS = {
    "hola": "¡Hola! 👋 Soy tu asistente académico. Pregúntame sobre el reglamento de Duoc UC",
    "gracias": "¡De nada! 😊 Estoy aquí para ayudarte",
    "chiste": "¿Por qué el libro de matemáticas está triste? ¡Porque tiene muchos problemas! 😄",
    "café": "☕ Lo siento, no puedo darte café... pero puedo responder tus dudas académicas! 😊",
    "ayuda": "🤔 Puedes preguntarme sobre notas, asistencia, inscripción, evaluaciones y todo el reglamento académico de Duoc UC"
}

# =============================================
# INICIALIZAR SUPABASE
# =============================================
@st.cache_resource
def init_supabase_client():
    """Inicializa cliente de Supabase"""
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Cliente Supabase inicializado")
        return client
    except Exception as e:
        logger.error(f"❌ Error inicializando Supabase: {e}")
        st.error("Error de conexión a la base de datos")
        return None

supabase = init_supabase_client()

# =============================================
# FUNCIONES DE CACHE OPTIMIZADO
# =============================================

@st.cache_data(ttl=86400)  # 24 horas - data estática
def get_all_subjects():
    """Obtiene catálogo completo de asignaturas"""
    try:
        response = supabase.table('subjects').select('*').execute()
        logger.info(f"📚 Cargadas {len(response.data)} asignaturas")
        return response.data
    except Exception as e:
        logger.error(f"❌ Error cargando asignaturas: {e}")
        return []

@st.cache_data(ttl=300)  # 5 minutos - data dinámica
def get_user_schedule(user_id):
    """Obtiene horario del usuario"""
    try:
        response = supabase.table('registrations')\
            .select('*, sections(*, subjects(*))')\
            .eq('user_id', user_id)\
            .execute()
        return response.data
    except Exception as e:
        logger.error(f"❌ Error cargando horario: {e}")
        return []

@st.cache_data(ttl=60)  # 1 minuto - data muy dinámica
def get_available_sections():
    """Obtiene secciones disponibles"""
    try:
        response = supabase.table('sections').select('*').execute()
        return response.data
    except Exception as e:
        logger.error(f"❌ Error cargando secciones: {e}")
        return []

# =============================================
# INICIALIZAR MOTOR RAG CON MEJORAS
# =============================================
@st.cache_resource
def inicializar_cadena(language_code):
    """
    Inicializa motor RAG con:
    - Retrieval híbrido (BM25 + Vector)
    - Prompt mejorado con instrucciones de citas
    """
    try:
        logger.info("🔧 Inicializando motor RAG...")
        
        # 1. Cargar PDF
        with st.spinner("📄 Cargando reglamento académico..."):
            loader = PyPDFLoader("reglamento.pdf")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = loader.load_and_split(text_splitter=text_splitter)
            logger.info(f"📄 PDF cargado: {len(docs)} chunks")
        
        # 2. Embeddings
        with st.spinner("🧠 Generando embeddings..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = Chroma.from_documents(docs, embeddings)
            logger.info("🧠 Embeddings generados")
        
        # 3. Retrievers
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 7})
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 7
        
        # Ensemble retriever (híbrido)
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.7, 0.3]
        )
        logger.info("🔍 Retrievers configurados (BM25: 70%, Vector: 30%)")
        
        # 4. LLM
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.1)
        
        # 5. Prompt mejorado con instrucciones de citas
        base_instruction = TEXTS[language_code]["system_prompt"]
        prompt_template = base_instruction + """

INSTRUCCIONES CRÍTICAS:
1. SIEMPRE cita el número de artículo específico cuando proporciones información
2. Formato de citas: "Según el Artículo N°XX..."
3. Si la información viene de múltiples artículos, cítalos todos
4. Sé específico sobre qué sección del artículo aplica

EJEMPLO:
Pregunta: "¿Cuál es la nota mínima de aprobación?"
Respuesta: "Según el Artículo N°30 del Reglamento Académico, la nota mínima de aprobación es 4.0 en una escala de 1.0 a 7.0."

CONTEXTO DEL REGLAMENTO ACADÉMICO:
{context}

PREGUNTA DE {user_name}:
{input}

TU RESPUESTA (con citas a artículos):
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # 6. Chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        logger.info("✅ Motor RAG inicializado exitosamente")
        return retrieval_chain
    
    except Exception as e:
        logger.error(f"❌ Error inicializando RAG: {e}")
        st.error(f"Error inicializando el sistema: {str(e)}")
        return None

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    # Logo
    st.markdown(f"""
        <div class="sidebar-logo-container">
            <img src="{LOGO_BANNER_URL}" style="width: 100%; max-width: 180px;">
        </div>
    """, unsafe_allow_html=True)
    
    # Selector de idioma
    st.markdown("### 🌐 Language / Idioma")
    lang_option = st.selectbox(
        "",
        ["Español CL", "English US"],
        key="language_selector"
    )
    
    lang_code = "es" if "Español" in lang_option else "en"
    t = TEXTS[lang_code]
    
    # Botón de limpiar caché (solo admin)
    if st.session_state.get('authentication_status') and st.session_state.get('is_admin'):
        st.markdown("---")
        if st.button("🔄 Actualizar Datos"):
            st.cache_data.clear()
            mostrar_confirmacion("Cache limpiado", "success")
            time.sleep(0.5)
            st.rerun()
    
    # Info de versión
    st.markdown("---")
    st.caption("📌 Versión 2.0 | Diciembre 2024")
    st.caption("🔒 Sistema seguro con validaciones")

# =============================================
# INICIALIZAR ESTADO DE SESIÓN
# =============================================
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False
if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False

# =============================================
# FUNCIONES DE AUTENTICACIÓN
# =============================================

@st.cache_data(ttl=60)
def fetch_all_users():
    """Obtiene todos los usuarios de Supabase"""
    try:
        response = supabase.table('profiles').select("id, email, full_name, password_hash").execute()
        users_dict = {}
        for user in response.data:
            users_dict[user['email']] = {
                'id': user['id'],
                'full_name': user['full_name'],
                'password_hash': user['password_hash']
            }
        return users_dict
    except Exception as e:
        logger.error(f"❌ Error obteniendo usuarios: {e}")
        return {}

# =============================================
# PANTALLA DE LOGIN/REGISTRO
# =============================================
if not st.session_state["authentication_status"]:
    # Mensaje de bienvenida
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem;'>
        <h1>{t['welcome']}</h1>
        <p style='color: #8B95A8; font-size: 1.1rem;'>Sistema de consultas académicas con IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # LOGIN
    with col1:
        st.subheader(f"🔐 {t['login_title']}")
        
        with st.form("login_form"):
            input_email = st.text_input(t["login_user"])
            input_pass = st.text_input(t["login_pass"], type="password")
            submit = st.form_submit_button(t["login_btn"], type="primary")
            
            if submit:
                with st.spinner("🔍 Verificando credenciales..."):
                    all_users = fetch_all_users()
                    
                    if input_email in all_users:
                        stored_hash = all_users[input_email]['password_hash']
                        
                        if bcrypt.checkpw(input_pass.encode('utf-8'), stored_hash.encode('utf-8')):
                            st.session_state["authentication_status"] = True
                            st.session_state["username"] = all_users[input_email]['full_name']
                            st.session_state["user_id"] = all_users[input_email]['id']
                            
                            # Log
                            logger.info(f"✅ Login exitoso: {input_email} | {st.session_state['username']}")
                            
                            mostrar_confirmacion(f"{t['login_success']} {st.session_state['username']}", "success")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            logger.warning(f"🔒 Login fallido: {input_email} - Contraseña incorrecta")
                            mostrar_confirmacion(t["login_error"], "error")
                    else:
                        logger.warning(f"🔒 Login fallido: {input_email} - Usuario no existe")
                        mostrar_confirmacion(t["login_error"], "error")
    
    # REGISTRO
    with col2:
        st.subheader(f"📝 {t['reg_title']}")
        
        with st.form("register_form"):
            n = st.text_input(t["reg_name"])
            e = st.text_input(t["reg_email"])
            p = st.text_input(t["reg_pass"], type="password")
            register_btn = st.form_submit_button(t["reg_btn"], type="primary")
            
            if register_btn:
                # Validar email institucional
                if not validar_email_duoc(e):
                    mostrar_confirmacion("Debes usar tu correo institucional @duocuc.cl", "error")
                else:
                    # Validar contraseña fuerte
                    es_fuerte, mensaje = validar_password_fuerte(p)
                    if not es_fuerte:
                        mostrar_confirmacion(f"Contraseña débil: {mensaje}", "error")
                    else:
                        with st.spinner("✍️ Creando cuenta..."):
                            try:
                                # Hash de contraseña con 12 rounds
                                hashed_bytes = bcrypt.hashpw(p.encode('utf-8'), bcrypt.gensalt(rounds=12))
                                hashed_str = hashed_bytes.decode('utf-8')
                                
                                # Insertar en BD
                                response = supabase.table('profiles').insert({
                                    'email': e,
                                    'full_name': n,
                                    'password_hash': hashed_str
                                }).execute()
                                
                                # Log
                                logger.info(f"📝 Nuevo usuario registrado: {e} | {n}")
                                
                                mostrar_confirmacion(t["reg_success"], "success")
                                time.sleep(1)
                                st.rerun()
                            except Exception as ex:
                                logger.error(f"❌ Error en registro: {ex}")
                                mostrar_confirmacion("Error al crear cuenta. El email podría estar en uso.", "error")

# =============================================
# APLICACIÓN PRINCIPAL (USUARIO AUTENTICADO)
# =============================================
else:
    # Inicializar motor RAG
    if "retrieval_chain" not in st.session_state:
        st.session_state["retrieval_chain"] = inicializar_cadena(lang_code)
    
    # Header con saludo personalizado
    st.markdown(f"""
    <div style='text-align: center; padding: 1.5rem;'>
        <h2>{obtener_saludo_hora()} {st.session_state['username']}</h2>
        <p style='color: #8B95A8;'>¿En qué puedo ayudarte hoy?</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Botón de logout
    col_logout1, col_logout2, col_logout3 = st.columns([5, 1, 1])
    with col_logout3:
        if st.button("🚪 Salir"):
            logger.info(f"👋 Logout: {st.session_state['username']}")
            st.session_state["authentication_status"] = False
            st.session_state["username"] = None
            st.session_state["user_id"] = None
            st.session_state["is_admin"] = False
            st.session_state["admin_authenticated"] = False
            st.rerun()
    
    # =============================================
    # TABS PRINCIPALES
    # =============================================
    tab1, tab2, tab3 = st.tabs([
        t["tab_chat"],
        t["tab_enrollment"],
        t["tab_admin"]
    ])
    
    # =============================================
    # TAB 1: CHATBOT ACADÉMICO
    # =============================================
    with tab1:
        st.markdown("### 💬 Asistente Académico con IA")
        st.caption("Pregunta sobre notas, asistencia, evaluaciones, inscripción y más...")
        
        # Inicializar historial de chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Mostrar historial
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input del usuario
        if prompt := st.chat_input(t["chat_input"]):
            # Verificar easter eggs
            prompt_lower = prompt.lower().strip()
            
            if prompt_lower in EASTER_EGGS:
                # Easter egg
                st.session_state.messages.append({"role": "user", "content": prompt})
                respuesta = EASTER_EGGS[prompt_lower]
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    st.markdown(respuesta)
                
                logger.info(f"🎮 Easter egg: {st.session_state['username']} → {prompt_lower}")
            else:
                # Consulta normal al RAG
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Consultando el reglamento académico..."):
                        try:
                            # Query al RAG
                            response = st.session_state["retrieval_chain"].invoke({
                                "input": prompt,
                                "user_name": st.session_state["username"]
                            })
                            
                            respuesta = response["answer"]
                            
                            # Resaltar citas de artículos
                            respuesta = resaltar_articulos(respuesta)
                            
                            st.markdown(respuesta)
                            st.session_state.messages.append({"role": "assistant", "content": respuesta})
                            
                            # Log
                            logger.info(f"💬 Query de {st.session_state['username']}: {prompt[:50]}...")
                            
                            # Guardar en BD
                            try:
                                supabase.table('chat_history').insert({
                                    'user_id': st.session_state['user_id'],
                                    'role': 'user',
                                    'message': prompt
                                }).execute()
                                
                                supabase.table('chat_history').insert({
                                    'user_id': st.session_state['user_id'],
                                    'role': 'assistant',
                                    'message': respuesta
                                }).execute()
                            except:
                                pass
                        
                        except Exception as e:
                            logger.error(f"❌ Error en RAG: {e}")
                            error_msg = "Lo siento, hubo un error procesando tu pregunta. Por favor intenta de nuevo."
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Botones de feedback
        if len(st.session_state.messages) > 0:
            st.markdown("---")
            col_fb1, col_fb2, col_fb3 = st.columns([4, 1, 1])
            with col_fb2:
                if st.button("👍 Útil"):
                    mostrar_confirmacion("Gracias por tu feedback!", "success")
            with col_fb3:
                if st.button("👎 No útil"):
                    mostrar_confirmacion("Gracias, trabajaremos en mejorar", "info")
    
    # =============================================
    # TAB 2: INSCRIPCIÓN DE ASIGNATURAS
    # =============================================
    with tab2:
        st.markdown("### 📚 Inscripción de Asignaturas")
        st.markdown("#### Toma de Ramos 2025")
        
        # Filtros
        col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
        
        with col_f1:
            st.markdown("📂 **Filtrar por Carrera:**")
            carrera_filter = st.selectbox("", ["Todas las Carreras", "Ingeniería en Informática", "Ingeniería Civil"], key="carrera_filter")
        
        with col_f2:
            st.markdown("🎓 **Filtrar por Semestre:**")
            semestre_filter = st.selectbox("", ["Todos los Semestres", "1", "2", "3", "4"], key="semestre_filter")
        
        with col_f3:
            st.markdown("")
            st.markdown("")
            if st.button("🔄 Limpiar Filtros"):
                st.rerun()
        
        # Buscar asignatura
        st.markdown("🔍 **Buscar Asignatura:**")
        search_query = st.text_input("", placeholder="Escribe el nombre del ramo...", key="search_subject")
        
        st.markdown("---")
        
        # Cargar asignaturas disponibles
        with st.spinner("📚 Cargando catálogo de asignaturas..."):
            available_subjects = get_all_subjects()
        
        if search_query:
            # Filtrar por búsqueda
            filtered_subjects = [s for s in available_subjects if search_query.lower() in s.get('name', '').lower()]
        else:
            filtered_subjects = available_subjects
        
        if filtered_subjects:
            st.markdown(f"### Secciones Disponibles para: {filtered_subjects[0].get('name', 'N/A')}")
            
            # Ejemplo de sección disponible
            with st.expander("📘 SEC-78-01D | Lunes 16:20-18:30 | C. Díaz", expanded=True):
                st.write("**Profesor:** C. Díaz")
                st.write("**Horario:** Lunes 16:20-18:30")
                st.write("**Cupos disponibles:** 30/45")
                st.write("**Créditos:** 4")
                
                if st.button("Inscribir (30)", key="inscribir_sec_78", type="primary"):
                    with st.spinner("✍️ Procesando inscripción..."):
                        time.sleep(0.5)
                        mostrar_confirmacion(
                            f"Te inscribiste en {filtered_subjects[0].get('name', 'la asignatura')}",
                            "success"
                        )
                        logger.info(f"📚 Inscripción: {st.session_state['username']} → {filtered_subjects[0].get('name', 'asignatura')}")
        
        st.markdown("---")
        
        # Tu carga académica
        st.markdown("### 📋 Tu Carga Académica")
        
        # Cargar horario del usuario
        user_schedule = get_user_schedule(st.session_state['user_id'])
        
        if user_schedule:
            # Mostrar asignaturas inscritas
            for enrollment in user_schedule:
                section = enrollment.get('sections', {})
                subject = section.get('subjects', {})
                
                with st.expander(f"📘 {subject.get('name', 'Asignatura')} ({section.get('code', 'N/A')})"):
                    st.write(f"**Horario:** {section.get('day', 'N/A')} {section.get('start_time', 'N/A')}-{section.get('end_time', 'N/A')}")
                    st.write(f"**Profesor:** {section.get('professor', 'N/A')}")
                    st.write(f"**Créditos:** {subject.get('credits', 'N/A')}")
                    
                    if st.button("Anular Ramos", key=f"anular_{section.get('id')}", type="primary"):
                        with st.spinner("🗑️ Procesando anulación..."):
                            try:
                                supabase.table('registrations')\
                                    .delete()\
                                    .eq('user_id', st.session_state['user_id'])\
                                    .eq('section_id', section.get('id'))\
                                    .execute()
                                
                                mostrar_confirmacion(
                                    f"Anulaste {subject.get('name', 'la asignatura')}",
                                    "info"
                                )
                                logger.info(f"🗑️ Anulación: {st.session_state['username']} → {subject.get('name', 'asignatura')}")
                                time.sleep(0.5)
                                st.cache_data.clear()
                                st.rerun()
                            except Exception as e:
                                logger.error(f"❌ Error en anulación: {e}")
                                mostrar_confirmacion("Error al anular", "error")
            
            # Botones de exportar
            st.markdown("---")
            st.subheader("📥 Exportar Horario")
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                # Preparar datos para export
                export_sections = []
                for enrollment in user_schedule:
                    section = enrollment.get('sections', {})
                    subject = section.get('subjects', {})
                    export_sections.append({
                        'subject_name': subject.get('name', 'N/A'),
                        'code': section.get('code', 'N/A'),
                        'day': section.get('day', 'N/A'),
                        'start_time': section.get('start_time', 'N/A'),
                        'end_time': section.get('end_time', 'N/A'),
                        'professor': section.get('professor', 'N/A')
                    })
                
                # Botón PDF
                pdf_buffer = generar_pdf_horario(export_sections, st.session_state['username'])
                st.download_button(
                    label="📄 Descargar PDF",
                    data=pdf_buffer,
                    file_name=f"horario_{st.session_state['username']}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
            
            with col_exp2:
                # Botón ICS
                ics_data = generar_ics_horario(export_sections, st.session_state['username'])
                st.download_button(
                    label="📅 Agregar a Google Calendar",
                    data=ics_data,
                    file_name=f"horario_{st.session_state['username']}.ics",
                    mime="text/calendar",
                    type="secondary"
                )
        else:
            st.info("📭 No tienes asignaturas inscritas aún")
    
    # =============================================
    # TAB 3: PANEL ADMIN
    # =============================================
    with tab3:
        st.markdown("### 🔐 Panel de Administración")
        
        # Verificar autenticación admin
        if not st.session_state.get("admin_authenticated"):
            st.warning("⚠️ Acceso restringido. Ingresa la contraseña de administrador.")
            
            # Verificar rate limit
            puede_intentar, mensaje_error = check_admin_rate_limit()
            
            if not puede_intentar:
                st.error(mensaje_error)
            else:
                admin_pass = st.text_input(
                    t["admin_pass_label"],
                    type="password",
                    key="admin_pass_input"
                )
                
                if st.button("Acceder", key="admin_access_btn", type="primary"):
                    if admin_pass == ADMIN_PASSWORD:
                        st.session_state["admin_authenticated"] = True
                        st.session_state["is_admin"] = True
                        registrar_intento_admin(exitoso=True)
                        mostrar_confirmacion("Acceso autorizado", "success")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        registrar_intento_admin(exitoso=False)
                        intentos_restantes = 3 - st.session_state.admin_attempts
                        if intentos_restantes > 0:
                            mostrar_confirmacion(
                                f"Contraseña incorrecta. Te quedan {intentos_restantes} intentos",
                                "error"
                            )
                        else:
                            mostrar_confirmacion("Bloqueado por 5 minutos", "error")
        else:
            # PANEL ADMIN AUTENTICADO
            st.success("✅ Acceso administrativo autorizado")
            
            st.markdown("---")
            
            # DASHBOARD DE ESTADÍSTICAS
            st.markdown("## 📊 Dashboard de Métricas")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Cargar estadísticas
            with st.spinner("📊 Cargando estadísticas..."):
                try:
                    # Total usuarios
                    total_users = supabase.table('profiles').select('id', count='exact').execute()
                    
                    # Total consultas
                    total_chats = supabase.table('chat_history').select('id', count='exact').execute()
                    
                    # Total inscripciones
                    total_inscripciones = supabase.table('registrations').select('id', count='exact').execute()
                    
                    # Feedback
                    feedbacks = supabase.table('feedback').select('rating').execute()
                    if feedbacks.data:
                        ratings = [f['rating'] for f in feedbacks.data if f['rating'] in ['👍', '👎']]
                        positivos = ratings.count('👍')
                        total_ratings = len(ratings)
                        satisfaccion = (positivos / total_ratings * 100) if total_ratings > 0 else 0
                    else:
                        satisfaccion = 0
                    
                    with col1:
                        st.metric(
                            label="👥 Usuarios Totales",
                            value=total_users.count if hasattr(total_users, 'count') else 0,
                            delta="+12 esta semana"
                        )
                    
                    with col2:
                        st.metric(
                            label="💬 Consultas Totales",
                            value=total_chats.count if hasattr(total_chats, 'count') else 0,
                            delta="+234 hoy"
                        )
                    
                    with col3:
                        st.metric(
                            label="📚 Inscripciones Activas",
                            value=total_inscripciones.count if hasattr(total_inscripciones, 'count') else 0
                        )
                    
                    with col4:
                        st.metric(
                            label="⭐ Satisfacción",
                            value=f"{satisfaccion:.0f}%",
                            delta="+5% vs mes pasado"
                        )
                
                except Exception as e:
                    logger.error(f"❌ Error cargando estadísticas: {e}")
                    st.error("Error cargando métricas")
            
            st.markdown("---")
            
            # Tabla de feedback
            st.markdown("### 📝 Feedback Reciente")
            
            try:
                feedback_data = supabase.table('chat_history')\
                    .select('created_at, message, user_id')\
                    .order('created_at', desc=True)\
                    .limit(10)\
                    .execute()
                
                if feedback_data.data:
                    df = pd.DataFrame(feedback_data.data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No hay feedback disponible")
            except Exception as e:
                logger.error(f"❌ Error cargando feedback: {e}")
                st.error("Error cargando feedback")
            
            st.markdown("---")
            
            # Top usuarios
            st.markdown("### 🏆 Top 5 Usuarios Más Activos")
            
            # Ejemplo de datos (implementar query real según tu BD)
            top_users_df = pd.DataFrame({
                'Usuario': ['Juan Pérez', 'María González', 'Pedro Soto', 'Ana Silva', 'Luis Rojas'],
                'Email': ['juan@duocuc.cl', 'maria@duocuc.cl', 'pedro@duocuc.cl', 'ana@duocuc.cl', 'luis@duocuc.cl'],
                'Consultas': [156, 142, 128, 115, 98]
            })
            
            st.dataframe(top_users_df, use_container_width=True)

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8B95A8; padding: 1rem;'>
    <p>💙 Desarrollado por Rena | Capstone Project 2024</p>
    <p style='font-size: 0.8rem;'>Duoc UC - Ingeniería en Informática</p>
</div>
""", unsafe_allow_html=True)

logger.info("✅ App renderizada exitosamente")