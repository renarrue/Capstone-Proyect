# Versión 34.2 (MASTER: Feedback Restaurado + Comentarios Negativos OK)
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
from datetime import datetime, timedelta, time as dt_time
import bcrypt
import pandas as pd
import logging
from io import BytesIO
import random

# --- LIBRERÍAS PARA EXPORTAR ---
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch
from icalendar import Calendar, Event as ICalEvent

# --- CONFIGURACIÓN DE LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

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

# --- CARGAR CSS ---
def load_css(file_name):
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_css = os.path.join(directorio_actual, file_name)
    try:
        with open(ruta_css) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("styles.css")

# --- EASTER EGGS ---
EASTER_EGGS = {
    "hola": "¡Hola! 👋 Soy tu asistente académico. ¿En qué te ayudo hoy?",
    "gracias": "¡De nada! 😊 Estoy aquí para apoyarte en tu semestre.",
    "chiste": "¿Por qué el libro de matemáticas está triste? ¡Porque tiene muchos problemas! 😅 (Perdón, soy un bot).",
    "tengo sueño": "¡Ánimo! ☕ Un café y a seguir. Recuerda que el esfuerzo vale la pena.",
    "quien eres": "Soy el Asistente Virtual de Duoc UC, entrenado para ayudarte con tus trámites y dudas académicas 🤖.",
    "chao": "¡Nos vemos! Que tengas un excelente día. 👋",
    "adios": "¡Hasta pronto! Mucho éxito en tus estudios."
}

def obtener_saludo_hora():
    hora = datetime.now().hour
    if 5 <= hora < 12: return "☀️ ¡Buenos días,"
    elif 12 <= hora < 19: return "👋 ¡Buenas tardes,"
    else: return "🌙 ¡Buenas noches,"

# --- DATOS DUROS DEL CALENDARIO ---
DATOS_CALENDARIO = """
RESUMEN OFICIAL DE FECHAS CLAVE 2026 (Usar esta información con prioridad):
1. PRIMER SEMESTRE:
   - Semana Cero (Inducción): Del 02 de Marzo al 07 de Marzo de 2026.
   - Inicio de Clases: Lunes 09 de Marzo de 2026.
   - Término de Clases: 21 de Julio de 2026.
   - Período de Exámenes: Del 06 de Julio al 21 de Julio de 2026.
   - Retiro de Asignaturas: Hasta el 11 de Abril de 2026.

2. SEGUNDO SEMESTRE:
   - Inicio de Clases: Lunes 10 de Agosto de 2026.
   - Término de Clases: 22 de Diciembre de 2026.
   - Período de Exámenes: Del 07 de Diciembre al 22 de Diciembre de 2026.
   - Retiro de Asignaturas: Hasta el 12 de Septiembre de 2026.

3. FERIADOS Y RECESOS:
   - Semana Santa: 03 y 04 de Abril.
   - Día del Trabajador: 01 de Mayo.
   - Glorias Navales: 21 de Mayo.
   - Vacaciones de Invierno (Receso): Del 24 de Julio al 08 de Agosto.
   - Fiestas Patrias: 18 y 19 de Septiembre.
"""

# --- DICCIONARIO DE TRADUCCIONES ---
TEXTS = {
    "es": {
        "label": "Español 🇨🇱",
        "title": "Asistente Académico Duoc UC",
        "sidebar_lang": "Idioma / Language",
        "login_success": "Usuario:",
        "logout_btn": "Cerrar Sesión",
        "tab1": "💬 Chatbot Académico",
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
        "chat_welcome": "¡Hola **{name}**! 👋 Soy tu asistente virtual. Puedo responder dudas del reglamento o ayudarte a buscar asignaturas.",
        "chat_welcome_clean": "¡Hola **{name}**! Historial archivado. ¿En qué más te ayudo?",
        "chat_placeholder": "Ej: ¿Cuándo empiezan las clases? o 'Buscar ramo Inglés'",
        "chat_thinking": "Procesando...",
        "feedback_thanks": "¡Gracias por tu feedback! 👍",
        "feedback_report_sent": "Reporte enviado.",
        "feedback_modal_title": "📝 Cuéntanos qué salió mal:",
        "feedback_modal_placeholder": "Ej: La fecha entregada es incorrecta, o la respuesta no tiene sentido...",
        "btn_send": "Enviar Reporte",
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
        "sug_header": "💡 **¿No sabes qué preguntar? Prueba con esto:**",
        "sug_btn1": "📅 Inicio de Clases",
        "sug_query1": "¿Cuándo comienzan las clases este semestre?",
        "sug_btn2": "🎓 Requisitos Titulación",
        "sug_query2": "¿Cuáles son los requisitos para titularme?",
        "sug_btn3": "🔍 Buscar Ramo",
        "sug_query3": "Quiero buscar la asignatura de Portafolio",
        "system_prompt": "INSTRUCCIÓN: Responde en Español formal pero cercano. ROL: Eres un coordinador académico de Duoc UC."
    },
    "en": {
        "label": "English 🇺🇸",
        "title": "Duoc UC Academic Assistant",
        "sidebar_lang": "Language / Idioma",
        "login_success": "User:",
        "logout_btn": "Log Out",
        "tab1": "💬 Academic Chat",
        "tab2": "📅 Course Enrollment",
        "tab3": "🔐 Admin / Audit",
        "login_title": "Student Login",
        "login_user": "Institutional Email",
        "login_pass": "Password",
        "login_btn": "Login",
        "login_failed": "❌ Invalid credentials",
        "login_welcome": "Welcome to the Assistant!",
        "chat_clear_btn": "🧹 Clear Conversation",
        "chat_cleaning": "Processing...",
        "chat_cleaned": "History cleared!",
        "chat_welcome": "Hello **{name}**! 👋 I'm your Duoc UC assistant. Ask me about rules, dates, or search for courses.",
        "chat_welcome_clean": "Hello **{name}**! History archived. Can I help with anything else?",
        "chat_placeholder": "Ex: When do classes start?",
        "chat_thinking": "Processing...",
        "feedback_thanks": "Thanks for your feedback! 👍",
        "feedback_report_sent": "Report sent.",
        "feedback_modal_title": "What went wrong?",
        "feedback_modal_placeholder": "Ex: The date is wrong...",
        "btn_send": "Send Comment",
        "btn_cancel": "Skip",
        "enroll_title": "Course Registration 2025",
        "filter_career": "📂 Filter by Career:",
        "filter_sem": "⏳ Filter by Semester:",
        "filter_all": "All Careers",
        "filter_all_m": "All Semesters",
        "reset_btn": "🔄 Clear Filters",
        "search_label": "📚 Search Subject:",
        "search_placeholder": "Type subject name...",
        "sec_title": "Available Sections for:",
        "btn_enroll": "Enroll",
        "btn_full": "Full",
        "msg_enrolled": "✅ Subject enrolled successfully!",
        "msg_conflict": "⛔ Error: Schedule Conflict",
        "msg_already": "ℹ️ You are already enrolled.",
        "my_schedule": "Your Academic Load",
        "no_schedule": "No subjects enrolled.",
        "btn_drop": "Drop Course",
        "msg_dropped": "Subject removed from load.",
        "admin_title": "Control Panel (Admin)",
        "admin_pass_label": "Access Key:",
        "admin_success": "Access Granted",
        "admin_info": "Log of interactions and negative feedback.",
        "admin_update_btn": "🔄 Refresh Data",
        "col_date": "Date/Time",
        "col_status": "Status",
        "col_q": "Student Question",
        "col_a": "AI Answer",
        "col_val": "Rate",
        "col_com": "Detail",
        "reg_header": "Create Student Account",
        "reg_name": "Full Name",
        "reg_email": "Duoc Email",
        "reg_pass": "Create Password",
        "reg_btn": "Register",
        "reg_success": "Account created! Please login.",
        "auth_error": "Check your credentials.",
        "sug_header": "💡 **Don't know what to ask? Try this:**",
        "sug_btn1": "📅 Class Start Date",
        "sug_query1": "When do classes start this semester?",
        "sug_btn2": "🎓 Graduation Reqs",
        "sug_query2": "What are the requirements for graduation?",
        "sug_btn3": "🔍 Search Course",
        "sug_query3": "I want to search for Portfolio subject",
        "system_prompt": "INSTRUCTION: Respond in English, formal but friendly. ROLE: You are an academic coordinator at Duoc UC."
    }
}

# --- CARGA DE CLAVES ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "DUOC2025")

if not GROQ_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Error: Faltan claves de API. Verifica tus Secrets.")
    st.stop()

# --- SUPABASE ---
@st.cache_resource
def init_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase_client()

# --- STREAMING ---
def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# --- FUNCIONES DE EXPORTACIÓN ---
def generar_pdf_horario(sections, user_name):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph(f"<b>Horario Académico - {user_name}</b>", styles['Title']))
    elements.append(Spacer(1, 0.3*inch))
    
    data = [['Asignatura', 'Sección', 'Día', 'Horario', 'Profesor']]
    for section in sections:
        subj = section.get('subjects', {})
        data.append([
            subj.get('name', 'N/A'),
            section.get('section_code', 'N/A'),
            section.get('day_of_week', 'N/A'),
            f"{section.get('start_time', '')[:5]}-{section.get('end_time', '')[:5]}",
            section.get('professor_name', 'N/A')
        ])
    
    table = Table(data, colWidths=[2*inch, 0.8*inch, 1*inch, 1.2*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002342')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

def generar_ics_horario(sections, user_name):
    cal = Calendar()
    cal.add('prodid', '-//DuocUC//Chatbot//')
    cal.add('version', '2.0')
    dias_map = {'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3, 'Viernes': 4, 'Sábado': 5}
    base_date = datetime(2026, 3, 9) 
    
    for section in sections:
        event = ICalEvent()
        subj = section.get('subjects', {}).get('name', 'Clase')
        event.add('summary', f"{subj}")
        dia_semana = section.get('day_of_week', 'Lunes')
        delta = dias_map.get(dia_semana, 0)
        dia_evento = base_date + timedelta(days=delta)
        start_str = section.get('start_time', '08:00:00')
        end_str = section.get('end_time', '10:00:00')
        start_dt = datetime.combine(dia_evento.date(), datetime.strptime(start_str, "%H:%M:%S").time())
        end_dt = datetime.combine(dia_evento.date(), datetime.strptime(end_str, "%H:%M:%S").time())
        event.add('dtstart', start_dt)
        event.add('dtend', end_dt)
        cal.add_component(event)
    return cal.to_ical()

# --- FUNCIÓN AGENTE: BUSCADOR DE RAMOS ---
def procesar_inscripcion_chat(texto_usuario, user_id):
    palabras_basura = ["inscribir", "toma de ramos", "tomar ramo", "buscar asignatura", "buscame el ramo", "quiero tomar", "buscar", "ramo", "asignatura", "el", "la", "de"]
    busqueda = texto_usuario.lower()
    for p in palabras_basura:
        busqueda = busqueda.replace(p, "")
    busqueda = busqueda.strip()
    
    if len(busqueda) < 3:
        return "Para buscar una asignatura, escribe su nombre específico (ej: 'Buscar Portafolio')."

    response = supabase.table('subjects').select('id, name, semester').ilike('name', f'%{busqueda}%').execute()
    asignaturas = response.data

    if not asignaturas:
        return f"🔍 No encontré asignaturas que coincidan con '{busqueda}'."

    respuesta = f"📚 **Resultados para '{busqueda}':**\n\n"
    hay_cupos = False
    
    for asig in asignaturas:
        respuesta += f"### {asig['name']} (Semestre {asig['semester']})\n"
        secciones = supabase.table('sections').select('*').eq('subject_id', asig['id']).execute().data
        
        if not secciones:
            respuesta += "_No hay secciones programadas._\n\n"
        else:
            for sec in secciones:
                inscritos = supabase.table('registrations').select('id', count='exact').eq('section_id', sec['id']).execute().count
                cupos = sec['capacity'] - (inscritos if inscritos else 0)
                estado_cupo = f"✅ **{cupos} cupos**" if cupos > 0 else "❌ Sin cupos"
                if cupos > 0: hay_cupos = True
                respuesta += f"- **Sección {sec['section_code']}**: {sec['day_of_week']} {sec['start_time'][:5]}-{sec['end_time'][:5]} | Prof: {sec['professor_name']} | {estado_cupo}\n"
            respuesta += "\n"

    if hay_cupos:
        respuesta += "\n👉 **Para inscribir, ve a la pestaña 'Inscripción de Asignaturas'.**"
    return respuesta

# --- CHATBOT ENGINE ---
@st.cache_resource
def inicializar_cadena(language_code):
    nombres_archivos = ["reglamento.pdf", "calendario_academico_2026.pdf"]
    base_path = os.path.dirname(os.path.abspath(__file__))
    all_docs = []
    
    for archivo in nombres_archivos:
        ruta_completa = os.path.join(base_path, archivo)
        try:
            loader = PyPDFLoader(ruta_completa)
            all_docs.extend(loader.load())
        except: continue

    if not all_docs:
        st.error("Error: No se encontraron documentos PDF.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs_procesados = text_splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs_procesados, embeddings)
    
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(docs_procesados)
    bm25_retriever.k = 5
    
    retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.7, 0.3])
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.1)
    
    base_instruction = TEXTS[language_code]["system_prompt"]
    
    prompt_template = base_instruction + f"""
    ROL: Asistente Académico experto.
    
    INFORMACIÓN OFICIAL OBLIGATORIA (CALENDARIO):
    {DATOS_CALENDARIO}

    INSTRUCCIONES DE RESPUESTA:
    1. Si preguntan por fechas, usa EXCLUSIVAMENTE los datos del calendario de arriba.
    2. Si preguntan por reglas (notas, asistencia), usa el contexto del Reglamento (abajo).
    3. REGLA DE LENGUAJE: Cuando hables de "Vacaciones de Invierno", NUNCA uses la palabra "suspenderán". Di "Las vacaciones SON del...".
    
    FIRMA: Despídete como "Tu Asistente Virtual Duoc UC".

    CONTEXTO ADICIONAL (PDFs):
    {{context}}
    
    PREGUNTA DE {{user_name}}: {{input}}
    RESPUESTA:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def fetch_all_users():
    try:
        response = supabase.table('profiles').select("email, full_name, password_hash").execute()
        if not response.data: return {}
        users_dict = {u['email']: u for u in response.data}
        return users_dict
    except: return {}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"""<div class="sidebar-logo-container"><img src="{LOGO_BANNER_URL}" style="width: 100%;"></div>""", unsafe_allow_html=True)
    lang_option = st.selectbox("🌐 Language / Idioma", ["Español 🇨🇱", "English 🇺🇸"])
    lang_code = "es" if "Español" in lang_option else "en"
    t = TEXTS[lang_code]

# --- CABECERA ---
col_title1, col_title2 = st.columns([0.1, 0.9])
with col_title1: st.image(LOGO_ICON_URL, width=70)
with col_title2: st.title(t["title"])

# --- AUTH STATE ---
if "authentication_status" not in st.session_state: st.session_state["authentication_status"] = None

# ==========================================
# APP PRINCIPAL
# ==========================================
if st.session_state["authentication_status"] is True:
    user_name = st.session_state["name"]
    user_email = st.session_state["username"]
    
    if 'user_id' not in st.session_state:
        user_id_response = supabase.table('profiles').select('id').eq('email', user_email).execute()
        if user_id_response.data: st.session_state.user_id = user_id_response.data[0]['id']
        else: st.stop()
    user_id = st.session_state.user_id

    c1, c2 = st.columns([0.8, 0.2])
    saludo_hora = obtener_saludo_hora()
    c1.subheader(f"{saludo_hora} {user_name}")
    c1.caption(f"Cuenta: {user_email}")
    
    if c2.button(t["logout_btn"], use_container_width=True):
        st.session_state["authentication_status"] = None
        st.session_state.clear()
        st.rerun()

    tab1, tab2, tab3 = st.tabs([t["tab1"], t["tab2"], t["tab3"]])

    # --- TAB 1: CHATBOT ---
    with tab1:
        if st.button("🧹 Limpiar Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        retrieval_chain = inicializar_cadena(lang_code)

        if "messages" not in st.session_state:
            st.session_state.messages = []
            history = supabase.table('chat_history').select('role, message').eq('user_id', user_id).order('created_at').execute()
            for row in history.data:
                st.session_state.messages.append({"role": row['role'], "content": row['message']})
            if not st.session_state.messages:
                st.session_state.messages.append({"role": "assistant", "content": f"¡Hola {user_name}! 👋"})

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # CHIPS
        if len(st.session_state.messages) < 2:
            st.markdown(t["sug_header"])
            c1, c2, c3 = st.columns(3)
            sugerencia = None
            if c1.button(t["sug_btn1"]): sugerencia = t["sug_query1"]
            if c2.button(t["sug_btn2"]): sugerencia = t["sug_query2"]
            if c3.button(t["sug_btn3"]): sugerencia = t["sug_query3"]
            
            if sugerencia:
                es_busqueda = "buscar" in sugerencia.lower() or "quiero" in sugerencia.lower()
                st.session_state.messages.append({"role": "user", "content": sugerencia})
                supabase.table('chat_history').insert({'user_id': user_id, 'role': 'user', 'message': sugerencia}).execute()
                
                with st.chat_message("assistant"):
                    with st.spinner(t["chat_thinking"]):
                        if es_busqueda:
                            resp = procesar_inscripcion_chat(sugerencia, user_id)
                        else:
                            resp = retrieval_chain.invoke({"input": sugerencia, "user_name": user_name})["answer"]
                    st.write(resp)
                
                st.session_state.messages.append({"role": "assistant", "content": resp})
                supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': resp}).execute()
                st.rerun()

        # INPUT CHAT CON EASTER EGGS & FEEDBACK COMPLETO
        if prompt := st.chat_input(t["chat_placeholder"]):
            # 1. Guardar mensaje usuario
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            supabase.table('chat_history').insert({'user_id': user_id, 'role': 'user', 'message': prompt}).execute()
            
            prompt_limpio = prompt.lower().strip().strip("?!.,")
            
            # 2. Generar respuesta
            with st.chat_message("assistant"):
                with st.spinner(t["chat_thinking"]):
                    if prompt_limpio in EASTER_EGGS:
                        resp = EASTER_EGGS[prompt_limpio]
                    elif any(x in prompt_limpio for x in ["inscribir", "toma de ramos", "buscar asignatura", "cupo", "buscar", "ramo"]):
                        resp = procesar_inscripcion_chat(prompt, user_id)
                    else:
                        try:
                            resp = retrieval_chain.invoke({"input": prompt, "user_name": user_name})["answer"]
                        except: resp = "Error de conexión."
                st.write_stream(stream_data(resp))
            
            # 3. Guardar respuesta asistente
            st.session_state.messages.append({"role": "assistant", "content": resp})
            supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': resp}).execute()

            # 4. LÓGICA DE FEEDBACK (RESTAURADO)
            with st.container():
                st.write("") 
                col_f1, col_f2, col_f3 = st.columns([0.1, 0.1, 0.8])
                unique_key = datetime.now().strftime("%H%M%S%f")
                
                # Estado para controlar la apertura del formulario
                feedback_key = f"fb_state_{unique_key}"

                with col_f1:
                    if st.button("👍", key=f"like_{unique_key}", help="Respuesta útil"):
                        supabase.table('feedback').insert({
                            'user_id': user_id,
                            'message': resp[:500],
                            'rating': 'good',
                            'comment': 'Voto positivo'
                        }).execute()
                        st.toast(t["feedback_thanks"])

                with col_f2:
                    if st.button("👎", key=f"dislike_{unique_key}", help="Reportar error"):
                        st.session_state[feedback_key] = True

                # Bloque condicional que MANTIENE el formulario abierto
                if st.session_state.get(feedback_key, False):
                    with st.expander(t["feedback_modal_title"], expanded=True):
                        with st.form(key=f"form_{unique_key}"):
                            comment = st.text_area(t["feedback_modal_placeholder"])
                            btn_send = st.form_submit_button(t["btn_send"])
                            
                            if btn_send:
                                supabase.table('feedback').insert({
                                    'user_id': user_id,
                                    'message': resp[:500],
                                    'rating': 'bad',
                                    'comment': comment if comment else "Sin detalle"
                                }).execute()
                                st.toast(t["feedback_report_sent"])
                                st.session_state[feedback_key] = False # Cerrar tras enviar
                                st.rerun()

    # --- TAB 2: INSCRIPCIÓN (CON FILTROS) ---
    with tab2:
        st.header(t["enroll_title"])
        
        @st.cache_data(ttl=60)
        def get_schedule(uid):
            return supabase.table('registrations').select('*, sections(*, subjects(*))').eq('user_id', uid).execute().data

        subjects_data = supabase.table('subjects').select('*').execute().data
        
        if subjects_data:
            # Filtros
            c_filter1, c_filter2, c_reset = st.columns([2, 2, 1])
            all_careers = sorted(list(set([s['career'] for s in subjects_data if s.get('career')])))
            all_semesters = sorted(list(set([s['semester'] for s in subjects_data if s.get('semester')])))
            
            with c_filter1: cat_carrera = st.selectbox("📂 Filtrar por Carrera", ["Todas"] + all_careers)
            with c_filter2: cat_semestre = st.selectbox("🎓 Filtrar por Semestre", ["Todos"] + [f"Semestre {s}" for s in all_semesters])
            with c_reset:
                st.write("") 
                st.write("") 
                if st.button("🔄 Limpiar"): pass

            filtered_subjects = subjects_data
            if cat_carrera != "Todas": filtered_subjects = [s for s in filtered_subjects if s['career'] == cat_carrera]
            if cat_semestre != "Todos": 
                sem_num = int(cat_semestre.split(" ")[1])
                filtered_subjects = [s for s in filtered_subjects if s['semester'] == sem_num]

            st.markdown("---")
            sel_name = st.selectbox(t["search_label"], [s['name'] for s in filtered_subjects], index=None, placeholder=f"Se encontraron {len(filtered_subjects)} asignaturas...")
            
            if sel_name:
                sid = next(s['id'] for s in subjects_data if s['name'] == sel_name)
                secs = supabase.table('sections').select('*').eq('subject_id', sid).execute().data
                
                if not secs: st.warning("No hay secciones programadas.")
                else:
                    for sec in secs:
                        with st.container(border=True):
                            inscritos = supabase.table('registrations').select('id', count='exact').eq('section_id', sec['id']).execute().count
                            cupos = sec['capacity'] - (inscritos if inscritos else 0)
                            col_a, col_b = st.columns([4, 1])
                            col_a.write(f"**{sec['section_code']}** | {sec['day_of_week']} | {sec['professor_name']}")
                            if cupos > 0:
                                if col_b.button(f"Inscribir ({cupos})", key=sec['id']):
                                    supabase.table('registrations').insert({'user_id': user_id, 'section_id': sec['id']}).execute()
                                    st.success("✅ Inscrito")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()
                            else:
                                col_b.button("Lleno", disabled=True, key=sec['id'])
        else:
            st.error("No se pudo cargar el catálogo.")

        st.divider()
        st.subheader("Tu Horario & Exportación")
        schedule = get_schedule(user_id)
        
        if schedule:
            c_exp1, c_exp2 = st.columns(2)
            sections_data = [r['sections'] for r in schedule]
            pdf_file = generar_pdf_horario(sections_data, user_name)
            c_exp1.download_button("📄 Descargar Horario PDF", pdf_file, "horario.pdf", "application/pdf", use_container_width=True)
            ics_file = generar_ics_horario(sections_data, user_name)
            c_exp2.download_button("📅 Exportar a Google Calendar", ics_file, "horario.ics", "text/calendar", use_container_width=True)

            for item in schedule:
                sec = item['sections']
                subj = sec['subjects']
                with st.expander(f"📘 {subj['name']}"):
                    c1, c2 = st.columns([4, 1])
                    c1.write(f"{sec['day_of_week']} {sec['start_time'][:5]}-{sec['end_time'][:5]} | {sec['professor_name']}")
                    if c2.button("Anular", key=f"del_{item['id']}"):
                        supabase.table('registrations').delete().eq('id', item['id']).execute()
                        st.rerun()
        else:
            st.info("No tienes ramos inscritos.")

    # --- TAB 3: ADMIN (DASHBOARD) ---
    with tab3:
        st.header(t["admin_title"])
        pwd = st.text_input("Clave Admin", type="password")
        if pwd == ADMIN_PASSWORD:
            st.success("Acceso Concedido")
            col1, col2, col3 = st.columns(3)
            total_users = supabase.table('profiles').select('id', count='exact', head=True).execute().count
            total_chats = supabase.table('chat_history').select('id', count='exact', head=True).execute().count
            feedbacks = supabase.table('feedback').select('rating').execute().data
            likes = len([f for f in feedbacks if f['rating'] == 'good'])
            col1.metric("Usuarios", total_users)
            col2.metric("Interacciones", total_chats)
            col3.metric("Likes", likes)
            st.subheader("Registro de Feedback")
            fb_data = supabase.table('feedback').select('*, profiles(email)').order('created_at', desc=True).execute().data
            if fb_data: st.dataframe(pd.DataFrame(fb_data))
            else: st.info("Sin feedback aún.")

# --- LOGIN FORM ---
else:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.subheader(t["login_title"])
        email = st.text_input(t["login_user"])
        pswd = st.text_input(t["login_pass"], type="password")
        if st.button(t["login_btn"], use_container_width=True):
            users = fetch_all_users()
            if email in users and bcrypt.checkpw(pswd.encode(), users[email]['password_hash'].encode()):
                st.session_state["authentication_status"] = True
                st.session_state["name"] = users[email]['full_name']
                st.session_state["username"] = email
                st.rerun()
            else:
                st.error("Credenciales incorrectas")
                
    with st.sidebar:
        st.subheader("Registro")
        n = st.text_input("Nombre")
        e = st.text_input("Email Duoc")
        p = st.text_input("Clave", type="password")
        if st.button("Crear Cuenta"):
            hashed = bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
            try:
                supabase.table('profiles').insert({'email': e, 'full_name': n, 'password_hash': hashed}).execute()
                st.success("¡Creado! Inicia sesión.")
            except: st.error("Error al crear usuario.")