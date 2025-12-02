# Versión 49.0 (FINAL: Fix Feedback API Error + ID Capture)
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
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
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
        "feedback_modal_placeholder": "Ej: La fecha entregada es incorrecta...",
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
        "sug_btn3": "📋 Justificar Inasistencia",
        "sug_query3": "¿Cómo justifico una inasistencia?",
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
        "sug_query1": "When do classes start?",
        "sug_btn2": "🎓 Graduation Reqs",
        "sug_query2": "Graduation requirements?",
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
    
    cell_style = ParagraphStyle('CellStyle', parent=styles['BodyText'], fontSize=9, leading=11, alignment=TA_CENTER, textColor=colors.whitesmoke)
    header_style = ParagraphStyle('HeaderStyle', parent=styles['BodyText'], fontSize=10, leading=12, alignment=TA_CENTER, textColor=colors.white, fontName='Helvetica-Bold')

    elements.append(Paragraph(f"<b>Horario Académico - {user_name}</b>", styles['Title']))
    elements.append(Spacer(1, 0.3*inch))
    
    headers = [Paragraph('Asignatura', header_style), Paragraph('Sección', header_style), Paragraph('Día', header_style), Paragraph('Horario', header_style), Paragraph('Profesor', header_style)]
    data = [headers]
    
    for section in sections:
        subj = section.get('subjects', {})
        data.append([
            Paragraph(subj.get('name', 'N/A'), cell_style),
            Paragraph(section.get('section_code', 'N/A'), cell_style),
            Paragraph(section.get('day_of_week', 'N/A'), cell_style),
            Paragraph(f"{section.get('start_time', '')[:5]}-{section.get('end_time', '')[:5]}", cell_style),
            Paragraph(section.get('professor_name', 'N/A'), cell_style)
        ])
    
    table = Table(data, colWidths=[2.5*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002342')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#003366')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
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
        delta = dias_map.get(section.get('day_of_week', 'Lunes'), 0)
        dia_evento = base_date + timedelta(days=delta)
        start_dt = datetime.combine(dia_evento.date(), datetime.strptime(section.get('start_time', '08:00:00'), "%H:%M:%S").time())
        end_dt = datetime.combine(dia_evento.date(), datetime.strptime(section.get('end_time', '10:00:00'), "%H:%M:%S").time())
        event.add('dtstart', start_dt)
        event.add('dtend', end_dt)
        cal.add_component(event)
    return cal.to_ical()

# --- CHATBOT ENGINE ---
@st.cache_resource
def inicializar_cadena(language_code):
    nombres_archivos = ["reglamento.pdf", "calendario_academico_2026.pdf"]
    base_path = os.path.dirname(os.path.abspath(__file__))
    all_docs = []
    for archivo in nombres_archivos:
        try: all_docs.extend(PyPDFLoader(os.path.join(base_path, archivo)).load())
        except: pass

    if not all_docs: st.error("Error: No se encontraron documentos PDF."); st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs_procesados = text_splitter.split_documents(all_docs)
    vector_store = Chroma.from_documents(docs_procesados, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    
    retriever = EnsembleRetriever(
        retrievers=[vector_store.as_retriever(search_kwargs={"k": 5}), BM25Retriever.from_documents(docs_procesados)],
        weights=[0.7, 0.3]
    )
    
    prompt = ChatPromptTemplate.from_template(TEXTS[language_code]["system_prompt"] + f"""
    ROL: Asistente Académico experto.
    DATOS CALENDARIO: {DATOS_CALENDARIO}
    REGLAS: 
    1. Usa el calendario para fechas. 
    2. Reglamento para normas.
    3. "Vacaciones" SON, no se suspenden.
    CONTEXTO: {{context}}
    PREGUNTA: {{input}}
    RESPUESTA:
    """)
    
    return create_retrieval_chain(retriever, create_stuff_documents_chain(ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.1), prompt))

# --- FETCH USERS ---
@st.cache_data(ttl=60)
def fetch_all_users():
    try:
        res = supabase.table('profiles').select("id, email, full_name, password_hash").execute()
        return {u['email']: u for u in res.data}
    except: return {}

# --- UI SIDEBAR ---
with st.sidebar:
    st.markdown(f"""<div class="sidebar-logo-container"><img src="{LOGO_BANNER_URL}" style="width: 100%;"></div>""", unsafe_allow_html=True)
    lang_option = st.selectbox("🌐 Language / Idioma", ["Español 🇨🇱", "English 🇺🇸"])
    lang_code = "es" if "Español" in lang_option else "en"
    t = TEXTS[lang_code]

# --- HEADER ---
c1, c2 = st.columns([0.1, 0.9])
with c1: st.image(LOGO_ICON_URL, width=70)
with c2: st.title(t["title"])

# --- AUTH STATE ---
if "authentication_status" not in st.session_state: st.session_state["authentication_status"] = None

if st.session_state["authentication_status"] is True:
    # VARIABLES DE SESION
    user_name = st.session_state["name"]
    user_email = st.session_state["username"]
    # FIX CRITICO PARA ID PERDIDO
    if 'user_id' not in st.session_state:
        try: st.session_state.user_id = supabase.table('profiles').select('id').eq('email', user_email).execute().data[0]['id']
        except: st.stop()
    user_id = st.session_state.user_id

    c1, c2 = st.columns([0.8, 0.2])
    c1.subheader(f"{obtener_saludo_hora()} {user_name}")
    c1.caption(f"{t['login_success']} {user_email}")
    if c2.button(t["logout_btn"]):
        st.session_state.clear(); st.rerun()

    tab1, tab2, tab3 = st.tabs([t["tab1"], t["tab2"], t["tab3"]])

    # --- TAB 1: CHATBOT ---
    with tab1:
        if st.button("🧹 Limpiar"): st.session_state.messages = []; st.rerun()
        st.divider()
        chain = inicializar_cadena(lang_code)
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": f"¡Hola {user_name}! 👋"}]
            
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if len(st.session_state.messages) < 2:
            st.markdown(t["sug_header"])
            c1, c2, c3 = st.columns(3)
            if c1.button(t["sug_btn1"]): prompt = t["sug_query1"]
            elif c2.button(t["sug_btn2"]): prompt = t["sug_query2"]
            elif c3.button(t["sug_btn3"]): prompt = t["sug_query3"]
            else: prompt = None
            
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                supabase.table('chat_history').insert({'user_id': user_id, 'role': 'user', 'message': prompt}).execute()
                st.rerun()

        if prompt := st.chat_input(t["chat_placeholder"]):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            supabase.table('chat_history').insert({'user_id': user_id, 'role': 'user', 'message': prompt}).execute()
            
            # Capturamos ID del mensaje insertado (si fuera necesario, pero para feedback usamos lógica nueva)
            
            prompt_limpio = prompt.lower().strip()
            with st.chat_message("assistant"):
                with st.spinner(t["chat_thinking"]):
                    if prompt_limpio in EASTER_EGGS: resp = EASTER_EGGS[prompt_limpio]
                    else:
                        try: resp = chain.invoke({"input": prompt, "user_name": user_name})["answer"]
                        except: resp = "Error de conexión."
                st.write(resp)
            
            # Guardamos respuesta del bot Y capturamos el objeto de respuesta para sacar el ID
            res_data = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': resp}).execute()
            # AQUÍ ESTA LA CLAVE: Guardamos el ID del mensaje recién creado
            if res_data.data:
                st.session_state.last_msg_id = res_data.data[0]['id']
            
            st.session_state.messages.append({"role": "assistant", "content": resp})

        # --- FEEDBACK CON FIX DE ID ---
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.write("---")
            if "fb_open" not in st.session_state: st.session_state.fb_open = False
            
            c1, c2, c3 = st.columns([0.1, 0.1, 0.8])
            # Usamos el ID capturado si existe, sino null (mejor que error)
            msg_id = st.session_state.get('last_msg_id', None)
            
            with c1:
                if st.button("👍", key=f"like_{int(time.time())}"):
                    if msg_id:
                        supabase.table('feedback').insert({'user_id': user_id, 'message_id': msg_id, 'rating': 'good'}).execute()
                        st.toast("Gracias!")
            
            with c2:
                if st.button("👎", key=f"dislike_{int(time.time())}"):
                    st.session_state.fb_open = True

            if st.session_state.fb_open:
                with st.container():
                    with st.form("fb_form"):
                        st.write(t["feedback_modal_title"])
                        txt = st.text_area(t["feedback_modal_placeholder"])
                        if st.form_submit_button(t["btn_send"]):
                            if msg_id:
                                # AQUI ESTABA EL ERROR: CAMBIADO 'message' POR 'message_id'
                                supabase.table('feedback').insert({
                                    'user_id': user_id, 
                                    'message_id': msg_id, 
                                    'rating': 'bad', 
                                    'comment': txt
                                }).execute()
                            st.toast("Enviado")
                            st.session_state.fb_open = False
                            st.rerun()
                        if st.form_submit_button(t["btn_cancel"]):
                            st.session_state.fb_open = False
                            st.rerun()

    # --- TAB 2: INSCRIPCIÓN ---
    with tab2:
        st.header(t["enroll_title"])
        subjects = supabase.table('subjects').select('*').execute().data
        if subjects:
            c1, c2, c3 = st.columns([2, 2, 1])
            cars = sorted(list(set(s['career'] for s in subjects if s['career'])))
            sems = sorted(list(set(s['semester'] for s in subjects if s['semester'])))
            
            with c1: f_car = st.selectbox(t["filter_career"], ["Todas"] + cars)
            with c2: f_sem = st.selectbox(t["filter_sem"], ["Todos"] + [f"Semestre {s}" for s in sems])
            with c3: 
                st.write(""); st.write("")
                if st.button(t["reset_btn"]): st.rerun()

            filtered = subjects
            if f_car != "Todas": filtered = [s for s in filtered if s['career'] == f_car]
            if f_sem != "Todos": filtered = [s for s in filtered if s['semester'] == int(f_sem.split(" ")[1])]

            st.markdown("---")
            sel = st.selectbox(t["search_label"], [s['name'] for s in filtered], index=None)
            
            if sel:
                sid = next(s['id'] for s in subjects if s['name'] == sel)
                secs = supabase.table('sections').select('*').eq('subject_id', sid).execute().data
                if not secs: st.warning("Sin secciones.")
                for s in secs:
                    with st.container(border=True):
                        insc = supabase.table('registrations').select('id', count='exact').eq('section_id', s['id']).execute().count
                        cupos = s['capacity'] - (insc or 0)
                        c_a, c_b = st.columns([4, 1])
                        c_a.write(f"**{s['section_code']}** | {s['day_of_week']} {s['start_time'][:5]} | {s['professor_name']}")
                        if cupos > 0:
                            if c_b.button(f"Inscribir ({cupos})", key=s['id']):
                                exist = supabase.table('registrations').select('id').eq('user_id', user_id).eq('section_id', s['id']).execute().data
                                if exist: st.warning("Ya inscrito.")
                                else:
                                    supabase.table('registrations').insert({'user_id': user_id, 'section_id': s['id']}).execute()
                                    st.success("Listo!"); st.cache_data.clear(); time.sleep(1); st.rerun()
                        else: c_b.button("Lleno", disabled=True)

        st.divider()
        st.subheader("Horario")
        sch = supabase.table('registrations').select('*, sections(*, subjects(*))').eq('user_id', user_id).execute().data
        if sch:
            c1, c2 = st.columns(2)
            secs_list = [r['sections'] for r in sch]
            c1.download_button("📄 PDF", generar_pdf_horario(secs_list, user_name), "h.pdf")
            c2.download_button("📅 ICS", generar_ics_horario(secs_list, user_name), "h.ics")
            for r in sch:
                s = r['sections']
                with st.expander(f"📘 {s['subjects']['name']}"):
                    c_a, c_b = st.columns([4, 1])
                    c_a.write(f"{s['day_of_week']} {s['start_time'][:5]} | {s['professor_name']}")
                    if c_b.button("Anular", key=f"del_{r['id']}"):
                        supabase.table('registrations').delete().eq('id', r['id']).execute(); st.rerun()
        else: st.info("Sin ramos.")

    # --- TAB 3: ADMIN ---
    with tab3:
        st.header(t["admin_title"])
        if not st.session_state.get('admin_auth'):
            p = st.text_input(t["admin_pass_label"], type="password")
            if st.button("Ingresar"):
                if p == ADMIN_PASSWORD: st.session_state.admin_auth = True; st.rerun()
                else: st.error("Error")
        else:
            st.success("Conectado")
            if st.button("Salir"): st.session_state.admin_auth = False; st.rerun()
            
            c1, c2, c3, c4 = st.columns(4)
            u_c = supabase.table('profiles').select('id', count='exact', head=True).execute().count
            c_c = supabase.table('chat_history').select('id', count='exact', head=True).execute().count
            fb = supabase.table('feedback').select('rating').execute().data
            likes = len([f for f in fb if f['rating'] == 'good'])
            dislikes = len([f for f in fb if f['rating'] == 'bad'])
            sat = int(likes/len(fb)*100) if fb else 0
            
            c1.metric("Usuarios", u_c)
            c2.metric("Chats", c_c)
            c3.metric("Satisfacción", f"{sat}%")
            c4.metric("Negativos", dislikes)
            
            if st.button("Refrescar"): st.rerun()
            
            st.subheader("Feedback")
            try:
                d = supabase.table('feedback').select('*, profiles(email)').order('created_at', desc=True).execute().data
                if d:
                    df = pd.DataFrame(d)
                    if 'profiles' in df.columns: df['Usuario'] = df['profiles'].apply(lambda x: x['email'] if x else '-')
                    st.dataframe(df.drop(columns=['profiles'], errors='ignore'))
                else: st.info("Vacío")
            except: pass

else:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader(t["login_title"])
        with st.form("login"):
            e = st.text_input("Email")
            p = st.text_input("Clave", type="password")
            if st.form_submit_button("Entrar"):
                fetch_all_users.clear()
                us = fetch_all_users()
                if e in us and bcrypt.checkpw(p.encode(), us[e]['password_hash'].encode()):
                    st.session_state.update({"authentication_status": True, "name": us[e]['full_name'], "username": e, "user_id": us[e]['id']})
                    st.rerun()
                else: st.error("Error")
    with c2:
        st.subheader("Registro")
        with st.form("reg"):
            n = st.text_input("Nombre")
            e_r = st.text_input("Email")
            p_r = st.text_input("Clave", type="password")
            if st.form_submit_button("Crear"):
                try:
                    h = bcrypt.hashpw(p_r.encode(), bcrypt.gensalt()).decode()
                    supabase.table('profiles').insert({'email': e_r, 'full_name': n, 'password_hash': h}).execute()
                    st.success("OK"); fetch_all_users.clear()
                except: st.error("Error")