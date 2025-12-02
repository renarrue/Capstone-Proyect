# Versión 36.0 (ESTABLE: Feedback Fix + RAG Puro + Tab Inscripción con Filtros)
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
import time
from datetime import datetime, timedelta, time as dt_time
import bcrypt
import pandas as pd
import logging
from io import BytesIO
import random

# --- LIBRERÍAS EXTRAS ---
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch
from icalendar import Calendar, Event as ICalEvent

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG ---
LOGO_ICON_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlve2kMlU53cq9Tl0DMxP0Ffo0JNap2dXq4q_uSdf4PyFZ9uraw7MU5irI6mA-HG8byNI&usqp=CAU"
LOGO_BANNER_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Logo_DuocUC.svg/2560px-Logo_DuocUC.svg.png"

st.set_page_config(page_title="Chatbot Duoc UC", page_icon=LOGO_ICON_URL, layout="wide")

# --- CSS ---
def load_css(file_name):
    try:
        with open(file_name) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except: pass

load_css("styles.css")

# --- DATOS DUROS CALENDARIO ---
DATOS_CALENDARIO = """
RESUMEN OFICIAL FECHAS 2026:
1. PRIMER SEMESTRE:
   - Semana Cero: 02 al 07 de Marzo 2026.
   - Inicio Clases: Lunes 09 de Marzo 2026.
   - Término Clases: 21 de Julio 2026.
   - Exámenes: 06 al 21 de Julio 2026.
   - Retiro Asignaturas: Hasta 11 de Abril 2026.

2. SEGUNDO SEMESTRE:
   - Inicio Clases: Lunes 10 de Agosto 2026.
   - Término Clases: 22 de Diciembre 2026.
   - Exámenes: 07 al 22 de Diciembre 2026.
   - Retiro Asignaturas: Hasta 12 de Septiembre 2026.

3. FERIADOS:
   - Semana Santa (03-04 Abril), Trabajador (01 Mayo), Glorias Navales (21 Mayo).
   - Vacaciones Invierno (Receso): 24 de Julio al 08 de Agosto.
   - Fiestas Patrias: 18-19 Septiembre.
"""

# --- TEXTOS ---
TEXTS = {
    "es": {
        "title": "Asistente Académico Duoc UC",
        "logout_btn": "Cerrar Sesión",
        "tab1": "💬 Chatbot Académico",
        "tab2": "📅 Inscripción de Asignaturas",
        "tab3": "🔐 Admin / Auditoría",
        "chat_placeholder": "Escribe tu duda aquí...",
        "sug_header": "💡 **Sugerencias:**",
        "sug_btn1": "📅 Inicio Clases", "sug_query1": "¿Cuándo empiezan las clases?",
        "sug_btn2": "🎓 Requisitos Titulación", "sug_query2": "¿Qué necesito para titularme?",
        "sug_btn3": "📋 Justificar Inasistencia", "sug_query3": "¿Cómo justifico una falta?",
        "system_prompt": "INSTRUCCIÓN: Responde en Español formal y cercano."
    },
    "en": {
        "title": "Duoc UC Assistant",
        "logout_btn": "Log Out",
        "tab1": "💬 Chatbot",
        "tab2": "📅 Enrollment",
        "tab3": "🔐 Admin",
        "chat_placeholder": "Type your question...",
        "sug_header": "💡 **Suggestions:**",
        "sug_btn1": "📅 Start Date", "sug_query1": "When do classes start?",
        "sug_btn2": "🎓 Graduation", "sug_query2": "Graduation requirements?",
        "sug_btn3": "📋 Absence", "sug_query3": "How to justify absence?",
        "system_prompt": "INSTRUCTION: Respond in English."
    }
}

# --- SECRETOS ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "DUOC2025")
except:
    st.error("Faltan credenciales en secrets.toml")
    st.stop()

# --- SUPABASE ---
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_supabase()

# --- RAG ENGINE ---
@st.cache_resource
def inicializar_cadena(lang):
    # Carga de Docs
    docs = []
    for f in ["reglamento.pdf", "calendario_academico_2026.pdf"]:
        path = os.path.join(os.path.dirname(__file__), f)
        try: docs.extend(PyPDFLoader(path).load())
        except: pass
    
    if not docs: return None

    # Split & Embeddings
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    vectorstore = Chroma.from_documents(splits, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    
    # Retrievers
    retriever = EnsembleRetriever(
        retrievers=[
            vectorstore.as_retriever(search_kwargs={"k": 5}),
            BM25Retriever.from_documents(splits)
        ],
        weights=[0.5, 0.5]
    )
    
    # Prompt
    template = TEXTS[lang]["system_prompt"] + f"""
    ROL: Coordinador Académico Duoc UC.
    
    DATOS CALENDARIO (USAR SIEMPRE PARA FECHAS):
    {DATOS_CALENDARIO}
    
    REGLAS:
    1. Usa el calendario de arriba para fechas.
    2. Usa el contexto abajo para reglas.
    3. NO digas "se suspenden" para vacaciones, di "son del...".
    
    CONTEXTO PDF: {{context}}
    PREGUNTA: {{input}}
    RESPUESTA:
    """
    
    chain = create_stuff_documents_chain(ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant"), ChatPromptTemplate.from_template(template))
    return create_retrieval_chain(retriever, chain)

# --- FUNCIONES AUX ---
def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

def get_users():
    res = supabase.table('profiles').select('email, full_name, password_hash').execute()
    return {u['email']: u for u in res.data}

# --- EXPORTACIÓN ---
def generar_pdf(sections, user):
    buff = BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=letter)
    elems = [Paragraph(f"<b>Horario - {user}</b>", getSampleStyleSheet()['Title']), Spacer(1, 0.2*inch)]
    data = [['Asignatura', 'Sección', 'Día', 'Horario']]
    for s in sections:
        data.append([s['subjects']['name'], s['section_code'], s['day_of_week'], f"{s['start_time'][:5]}-{s['end_time'][:5]}"])
    t = Table(data)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.navy),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),1,colors.black)]))
    elems.append(t)
    doc.build(elems)
    buff.seek(0)
    return buff

def generar_ics(sections):
    c = Calendar()
    for s in sections:
        e = ICalEvent()
        e.add('summary', s['subjects']['name'])
        c.add_component(e)
    return c.to_ical()

# --- UI ---
with st.sidebar:
    st.image(LOGO_BANNER_URL)
    lang_opt = st.selectbox("Idioma", ["Español", "English"])
    lang = "es" if "Español" in lang_opt else "en"
    t = TEXTS[lang]

# --- LOGIN ---
if "auth" not in st.session_state: st.session_state.auth = False

if not st.session_state.auth:
    st.title(t["title"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Ingresar")
        email = st.text_input("Email")
        pwd = st.text_input("Clave", type="password")
        if st.button("Entrar"):
            users = get_users()
            if email in users and bcrypt.checkpw(pwd.encode(), users[email]['password_hash'].encode()):
                st.session_state.auth = True
                st.session_state.user = users[email]
                st.rerun()
            else: st.error("Datos incorrectos")
    with c2:
        st.subheader("Registrar")
        n_reg = st.text_input("Nombre")
        e_reg = st.text_input("Email Nuevo")
        p_reg = st.text_input("Clave Nueva", type="password")
        if st.button("Crear"):
            try:
                h = bcrypt.hashpw(p_reg.encode(), bcrypt.gensalt()).decode()
                supabase.table('profiles').insert({'email':e_reg, 'full_name':n_reg, 'password_hash':h}).execute()
                st.success("Creado!")
            except: st.error("Error registro")

else:
    # --- APP ---
    u = st.session_state.user
    c1, c2 = st.columns([0.8, 0.2])
    c1.title(f"Hola, {u['full_name']}")
    if c2.button(t["logout_btn"]):
        st.session_state.auth = False
        st.rerun()

    tab1, tab2, tab3 = st.tabs([t["tab1"], t["tab2"], t["tab3"]])

    # TAB 1: CHATBOT (LIMPIO)
    with tab1:
        if st.button("🧹 Limpiar"):
            st.session_state.msgs = []
            st.rerun()

        if "msgs" not in st.session_state: st.session_state.msgs = []
        
        for m in st.session_state.msgs:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        # Chips
        if not st.session_state.msgs:
            cols = st.columns(3)
            if cols[0].button(t["sug_btn1"]): prompt_chip = t["sug_query1"]
            elif cols[1].button(t["sug_btn2"]): prompt_chip = t["sug_query2"]
            elif cols[2].button(t["sug_btn3"]): prompt_chip = t["sug_query3"]
            else: prompt_chip = None
            
            if prompt_chip:
                st.session_state.msgs.append({"role":"user", "content":prompt_chip})
                st.rerun()

        # Input
        if prompt := st.chat_input(t["chat_placeholder"]):
            st.session_state.msgs.append({"role":"user", "content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            # RAG Generación
            chain = inicializar_cadena(lang)
            if chain:
                with st.chat_message("assistant"):
                    with st.spinner("..."):
                        res = chain.invoke({"input":prompt, "user_name":u['full_name']})["answer"]
                        st.write_stream(stream_text(res))
                st.session_state.msgs.append({"role":"assistant", "content":res})
                
                # Guardar historial
                supabase.table('chat_history').insert({'user_id': u['id'] or 1, 'role':'user', 'message':prompt}).execute()
                supabase.table('chat_history').insert({'user_id': u['id'] or 1, 'role':'assistant', 'message':res}).execute()
            else:
                st.error("Error RAG")

        # --- FEEDBACK PERSISTENTE (FIX RESTAURADO) ---
        # Solo mostramos feedback si hay un mensaje del asistente reciente
        if st.session_state.msgs and st.session_state.msgs[-1]["role"] == "assistant":
            st.write("---")
            cf1, cf2 = st.columns([0.1, 0.9])
            
            # Estado del formulario
            if "fb_open" not in st.session_state: st.session_state.fb_open = False
            
            with cf1:
                if st.button("👍", key="fb_up"):
                    supabase.table('feedback').insert({'user_id': u['id'], 'message': st.session_state.msgs[-1]["content"][:200], 'rating':'good'}).execute()
                    st.toast("Gracias!")
            
            with cf2:
                if st.button("👎", key="fb_down"):
                    st.session_state.fb_open = True # Activa formulario
            
            # Formulario condicional que NO desaparece hasta enviar
            if st.session_state.fb_open:
                with st.form("fb_bad_form"):
                    st.caption("¿Qué salió mal?")
                    txt = st.text_area("Detalle")
                    if st.form_submit_button("Enviar"):
                        supabase.table('feedback').insert({'user_id': u['id'], 'message': st.session_state.msgs[-1]["content"][:200], 'rating':'bad', 'comment':txt}).execute()
                        st.toast("Reporte enviado")
                        st.session_state.fb_open = False # Cierra solo al enviar
                        st.rerun()


    # TAB 2: INSCRIPCIÓN (CON FILTROS RESTAURADOS)
    with tab2:
        subjects = supabase.table('subjects').select('*').execute().data
        if subjects:
            c1, c2 = st.columns(2)
            careers = sorted(list(set(s['career'] for s in subjects if s['career'])))
            sems = sorted(list(set(s['semester'] for s in subjects if s['semester'])))
            
            filtro_car = c1.selectbox("Carrera", ["Todas"] + careers)
            filtro_sem = c2.selectbox("Semestre", ["Todos"] + sems)
            
            # Lógica de filtrado
            lista_final = subjects
            if filtro_car != "Todas": lista_final = [x for x in lista_final if x['career'] == filtro_car]
            if filtro_sem != "Todos": lista_final = [x for x in lista_final if x['semester'] == filtro_sem]
            
            ramo = st.selectbox("Selecciona Asignatura", [x['name'] for x in lista_final], index=None)
            
            if ramo:
                sid = next(x['id'] for x in lista_final if x['name'] == ramo)
                secs = supabase.table('sections').select('*').eq('subject_id', sid).execute().data
                for sec in secs:
                    with st.container(border=True):
                        insc = supabase.table('registrations').select('id', count='exact').eq('section_id', sec['id']).execute().count
                        cupos = sec['capacity'] - (insc or 0)
                        c_a, c_b = st.columns([4,1])
                        c_a.write(f"**{sec['section_code']}** | {sec['day_of_week']} {sec['start_time'][:5]} | {sec['professor_name']}")
                        if cupos > 0:
                            if c_b.button(f"Tomar ({cupos})", key=sec['id']):
                                supabase.table('registrations').insert({'user_id':u['id'], 'section_id':sec['id']}).execute()
                                st.success("Inscrito!")
                                st.rerun()
                        else: c_b.button("Lleno", disabled=True)

        # Horario
        st.divider()
        sch = supabase.table('registrations').select('*, sections(*, subjects(*))').eq('user_id', u['id']).execute().data
        if sch:
            # Botones Export
            sec_list = [x['sections'] for x in sch]
            st.download_button("PDF", generar_pdf(sec_list, u['full_name']), "horario.pdf")
            st.download_button("Calendario", generar_ics(sec_list), "horario.ics")
            
            for r in sch:
                s = r['sections']
                with st.expander(f"{s['subjects']['name']}"):
                    st.write(f"{s['day_of_week']} {s['start_time']}")
                    if st.button("Borrar", key=f"del_{r['id']}"):
                        supabase.table('registrations').delete().eq('id', r['id']).execute()
                        st.rerun()

    # TAB 3: ADMIN
    with tab3:
        if st.text_input("Clave", type="password") == ADMIN_PASSWORD:
            st.success("OK")
            cnt_users = supabase.table('profiles').select('id', count='exact', head=True).execute().count
            cnt_msg = supabase.table('chat_history').select('id', count='exact', head=True).execute().count
            c1, c2 = st.columns(2)
            c1.metric("Usuarios", cnt_users)
            c2.metric("Mensajes", cnt_msg)
            
            st.subheader("Feedback Negativo")
            bad_fb = supabase.table('feedback').select('*').eq('rating', 'bad').execute().data
            st.dataframe(bad_fb)