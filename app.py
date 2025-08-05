import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
import rdflib
from collections import defaultdict
import pandas as pd

# Load .env jika ada
load_dotenv()

# ==================== Helper OG-RAG ====================
def uri_to_label(uri):
    uri_str = str(uri)
    return uri_str.split('#')[-1] if '#' in uri_str else uri_str.split('/')[-1]

def construct_hypergraph_from_rdf(rdf_path):
    g = rdflib.Graph()
    g.parse(rdf_path, format='xml')
    subj_map = defaultdict(lambda: defaultdict(list))
    for s, p, o in g:
        s_label = uri_to_label(s)
        p_label = uri_to_label(p)
        o_label = str(o) if isinstance(o, rdflib.Literal) else uri_to_label(o)
        subj_map[s_label][p_label].append(o_label)
    hyperedges = []
    for i, (subj, rels) in enumerate(subj_map.items()):
        nodes = []
        for pred, objs in rels.items():
            for j, obj in enumerate(objs):
                nodes.append({"node_id": f"{i}-{j}", "text": f"{subj} {pred} {obj}", "parent_subject": subj})
        hyperedges.append({
            "edge_id": i,
            "subject": subj,
            "relations": dict(rels),
            "nodes": nodes
        })
    return hyperedges

def create_faiss_index(hyperedges, embedding_model):
    all_docs = []
    for edge in hyperedges:
        for node in edge['nodes']:
            doc = Document(
                page_content=node['text'],
                metadata={"subject": node['parent_subject']}
            )
            all_docs.append(doc)
    faiss_index = FAISS.from_documents(documents=all_docs, embedding=embedding_model)
    return faiss_index

def build_context_from_hyperedges(matched_edges):
    if not matched_edges: return "Tidak ada konteks relevan yang ditemukan."
    context_str = ""
    for edge in matched_edges:
        context_str += f"Fakta terkait: {edge['subject']}\n"
        for relation, objects in edge['relations'].items():
            for obj in objects:
                context_str += f"- {relation}: {obj}\n"
        context_str += "\n"
    return context_str.strip()

def retrieve_context_from_faiss_with_scores(query, faiss_index, all_hyperedges, top_k=3):
    retrieved_docs_with_scores = faiss_index.similarity_search_with_score(query, k=top_k)
    retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]
    relevant_subjects = list(dict.fromkeys([doc.metadata['subject'] for doc in retrieved_docs]))
    subject_to_hyperedge = {edge['subject']: edge for edge in all_hyperedges}
    matched_hyperedges = [subject_to_hyperedge[subj] for subj in relevant_subjects if subj in subject_to_hyperedge]
    context = build_context_from_hyperedges(matched_hyperedges)
    return context, retrieved_docs_with_scores

def load_csv_docs(csv_url):
    df = pd.read_csv(csv_url)
    def row_to_doc(row):
        chunks = []
        for col in df.columns:
            if pd.notnull(row[col]):
                chunks.append(f"{col.capitalize()}: {row[col]}")
        return ". ".join(chunks)
    documents = []
    for idx, row in df.iterrows():
        content = row_to_doc(row)
        doc = Document(page_content=content, metadata={"row": idx})
        documents.append(doc)
    return documents

# ==================== UI CONFIG & STYLE ====================
st.set_page_config(page_title="Chatbot OG-RAG Medis", layout="wide")
st.markdown("""
<style>
    .chat-container {margin-bottom: 2rem;}
    .chat-bubble {padding: 0.98rem 1.15rem; border-radius: 16px; margin-bottom: 1.1rem; max-width: 85%; word-break: break-word;}
    .chat-user {
        background: linear-gradient(120deg, #aee1fb 20%, #d2f7ed 80%);
        color: #103649 !important;
        align-self: flex-end;
        font-weight: 500;
        border-bottom-right-radius: 2px;
    }
    .chat-ai {
        background: linear-gradient(120deg, #ebfaf4 20%, #e7e9f5 100%);
        color: #204350 !important;
        align-self: flex-start;
        border-bottom-left-radius: 2px;
        font-weight: 400;
    }
    .chat-meta {font-size: 0.83rem; color: #587a8a; margin-bottom: 0.28rem;}
    .stTextInput > div > div > input {font-size: 1.13rem;}
</style>
""", unsafe_allow_html=True)

st.title("💬 Chatbot OG-RAG Medis")
st.caption("Percakapan akan terekam pada session dan bisa diekspor ke CSV (fitur opsional).")

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    model_choice = st.selectbox("Pilih Model OpenAI:", ["gpt-4o", "gpt-4o-mini"])
    temperature_value = st.slider("Temperature (0 = deterministik, 1 = kreatif)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    method_choice = st.selectbox(
        "Pilih Metode:",
        ["Model Dasar (Tanpa RAG)", "RAG Standar", "Ontology-Grounded RAG (OGRAG)"]
    )
    openai_api_key = st.text_input("Masukkan OpenAI API Key Anda", type="password")
    st.markdown("---")
    st.info("Riwayat chat akan direset jika model/metode diganti.")

# ========== PATH BASE ==========
CSV_URL = "punya RAG.csv"
RDF_PATH = "Ontology Alodog tanpa peringatan.rdf"

# ========== MEMORY ==========
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'last_method' not in st.session_state: st.session_state.last_method = method_choice
if 'last_model' not in st.session_state: st.session_state.last_model = model_choice
if (st.session_state.last_method != method_choice) or (st.session_state.last_model != model_choice):
    st.session_state.chat_history = []
    st.session_state.last_method = method_choice
    st.session_state.last_model = model_choice

# ========== CHAIN SETUP ==========
@st.cache_resource
def setup_chain(method, model_name, api_key, temperature):
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    if method == "Model Dasar (Tanpa RAG)":
        prompt_template = "Anda adalah asisten AI medis. Jawab pertanyaan berikut dengan akurat: {question}"
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        return chain
    elif method == "RAG Standar":
        documents = load_csv_docs(CSV_URL)
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
        prompt_template_rag = """
Anda adalah asisten medis yang hanya boleh menggunakan informasi berikut (konteks) untuk menjawab pertanyaan. Jangan gunakan pengetahuan luar.
Jawab dengan bahasa inggris
Konteks:
{context}

Pertanyaan:
{question}

Jawaban:
"""
        prompt = PromptTemplate.from_template(prompt_template_rag)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()} |
            prompt |
            llm |
            StrOutputParser()
        )
        return rag_chain
    elif method == "Ontology-Grounded RAG (OGRAG)":
        @st.cache_resource
        def cache_hypergraph_and_faiss():
            hyperedges = construct_hypergraph_from_rdf(RDF_PATH)
            faiss_index = create_faiss_index(hyperedges, embeddings)
            return hyperedges, faiss_index
        hyperedges, faiss_index = cache_hypergraph_and_faiss()
        prompt_template_ograg = """
Anda adalah asisten AI medis yang akurat. Jawab *hanya* menggunakan fakta dalam "Konteks". Jika tidak ada, jangan tambahkan apapun. Jangan berimprovisasi.
Jika informasi tidak ada dalam konteks, jawab dengan "Informasi tidak ditemukan dalam konteks yang diberikan."

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:
"""
        prompt = PromptTemplate.from_template(prompt_template_ograg)
        def chain_ograg_faiss(user_question):
            context, _ = retrieve_context_from_faiss_with_scores(
                user_question, faiss_index, hyperedges, top_k=5
            )
            prompt_input = prompt.format(context=context, question=user_question)
            answer_obj = llm.invoke(prompt_input)
            answer = answer_obj.content if hasattr(answer_obj, "content") else str(answer_obj)
            return answer
        return chain_ograg_faiss

if openai_api_key:
    with st.spinner(f"Menyiapkan sistem ({method_choice})..."):
        chain = setup_chain(method_choice, model_choice, openai_api_key, temperature_value)
    st.success("✅ Sistem siap digunakan!")
else:
    st.warning("Masukkan OpenAI API key Anda di sidebar.")

# ========== Chat Bubble UI ==========
def render_chat_history():
    for entry in st.session_state.chat_history:
        user_msg, ai_msg = entry['user'], entry['ai']
        st.markdown(
            f'<div class="chat-container">'
            f'<div class="chat-bubble chat-user"><span class="chat-meta">Anda:</span><br>{user_msg}</div>'
            f'<div class="chat-bubble chat-ai"><span class="chat-meta">AI:</span><br>{ai_msg}</div>'
            f'</div>', unsafe_allow_html=True
        )

st.markdown("---")
render_chat_history()

# ========== User Input ==========
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ketik pertanyaan medis Anda...", key="input_text", autocomplete="off")
    submit_btn = st.form_submit_button("Kirim")

if submit_btn and user_input and openai_api_key:
    with st.spinner("AI sedang merespons..."):
        # Gabungkan riwayat percakapan ke prompt baru
        MAX_HISTORY = 3   # Misal, hanya 3 percakapan terakhir
        history_context = ""
        for h in st.session_state.chat_history[-MAX_HISTORY:]:
            history_context += f"User: {h['user']}\nAI: {h['ai']}\n"
        full_prompt = f"Riwayat percakapan:\n{history_context}\nUser: {user_input}\n"
        
        if method_choice == "Model Dasar (Tanpa RAG)":
            answer = chain.invoke({"question": full_prompt})
        elif method_choice == "Ontology-Grounded RAG (OGRAG)":
            answer = chain(full_prompt)
        else:
            answer = chain.invoke(full_prompt)

    st.session_state.chat_history.append({'user': user_input, 'ai': answer})
    st.rerun()
elif submit_btn and not openai_api_key:
    st.error("Masukkan API key OpenAI Anda terlebih dahulu.")

# ========== Export Chat ==========
with st.expander("📄 Ekspor Riwayat Chat"):
    if st.session_state.chat_history:
        df_chat = pd.DataFrame(st.session_state.chat_history)
        st.download_button(
            label="Unduh Riwayat Chat ke CSV",
            data=df_chat.to_csv(index=False),
            file_name="riwayat_chatbot_ograg.csv",
            mime="text/csv"
        )
    else:
        st.info("Belum ada percakapan untuk diekspor.")
