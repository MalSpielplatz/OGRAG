import streamlit as st
import os
from dotenv import load_dotenv
# BENAR (Cara Import)
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
import rdflib
from collections import defaultdict
import json

# Load environment variables from .env
load_dotenv()

# Helper Functions
def uri_to_label(uri):
    uri_str = str(uri)
    return uri_str.split('#')[-1] if '#' in uri_str else uri_str.split('/')[-1]

def build_hypergraph_from_rdf(rdf_path):
    g = rdflib.Graph()
    g.parse(rdf_path, format='xml')
    hyperedge_dict = defaultdict(list)
    for s, p, o in g:
        s_l, p_l, o_l = uri_to_label(s), uri_to_label(p), uri_to_label(o)
        if p_l == 'type' and o_l == 'NamedIndividual':
            continue
        hyperedge_dict[s_l].append((p_l, o_l))
    hyperedges = {f"edge_{i}": dict(facts) for i, (subj, facts) in enumerate(hyperedge_dict.items())}
    for subj_label, edge_id in zip(hyperedge_dict.keys(), hyperedges.keys()):
        hyperedges[edge_id]['subjek'] = subj_label
    return hyperedges

def hyperedges_to_docs(hyperedges):
    documents = []
    for edge_id, facts in hyperedges.items():
        content = json.dumps(facts, ensure_ascii=False, indent=2)
        doc = Document(page_content=content, metadata={"source": facts.get("subjek", "N/A")})
        documents.append(doc)
    return documents

# Streamlit App
st.set_page_config(page_title="Peningkatan Akurasi AI dengan RAG", layout="wide")

# --- PERUBAHAN JUDUL & NARASI ---
# DIUBAH: Judul diubah menjadi lebih positif dan fokus pada peningkatan.
st.title("🔬 Peningkatan Akurasi AI dengan Ontology-Grounded RAG")
st.markdown("Studi kasus untuk mengurangi halusinasi dan meningkatkan presisi jawaban AI menggunakan basis pengetahuan terstruktur (Ontologi).")


# Sidebar Configuration
st.sidebar.header("⚙️ Konfigurasi")

model_choice = st.sidebar.selectbox("Pilih Model OpenAI:", ["gpt-4o", "gpt-4o-mini"]) # DIUBAH: "Model GPT" -> "Model OpenAI"

# DIUBAH: Opsi diubah dari "RAG Biasa" menjadi "RAG Standar" agar lebih netral.
method_choice = st.sidebar.selectbox(
    "Pilih Metode Retrieval:",
    ["Ontology-Grounded RAG", "RAG Standar"]
)

openai_api_key = st.sidebar.text_input("Masukkan OpenAI API Key", type="password")

# Initialize session state
if 'rag_chain' not in st.session_state:
    st.session_state['rag_chain'] = None

# Load and Setup RAG
@st.cache_resource
def setup_rag(method, model_name, api_key):
    os.environ["OPENAI_API_KEY"] = api_key

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model=model_name, temperature=0)

    # --- PERUBAHAN PROMPT ---
    # DIUBAH: Prompt dibuat lebih tegas dan jelas untuk mengontrol output.
    prompt_template = """
    Anda adalah asisten AI medis yang cermat dan akurat.
    Gunakan HANYA informasi dari 'Konteks' di bawah untuk menjawab 'Pertanyaan'.
    Jangan sekali-kali menggunakan pengetahuan eksternal Anda.

    Konteks:
    {context}

    Pertanyaan:
    {question}

    Aturan Jawaban:
    1. Jika informasi untuk menjawab pertanyaan ADA di dalam konteks, berikan jawaban langsung berdasarkan informasi tersebut.
    2. Jika informasi TIDAK ADA di dalam konteks, jawab HANYA dengan kalimat: "Informasi tidak ditemukan dalam basis data."
    """

    prompt = PromptTemplate.from_template(prompt_template)

    if method == "Ontology-Grounded RAG":
        rdf_path = "Ontology Alodog tanpa peringatan.rdf"  # sesuaikan dengan path RDF lokal Anda
        hyperedges = build_hypergraph_from_rdf(rdf_path)
        documents = hyperedges_to_docs(hyperedges)
    else:  # RAG Standar dengan contoh data statis
        # DIUBAH: Deskripsi diubah untuk kejelasan.
        documents = [Document(page_content="Paracetamol adalah obat untuk meredakan demam dan nyeri ringan. Efek samping yang mungkin terjadi adalah mual atau insomnia.")]

    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} |
        prompt |
        llm |
        StrOutputParser()
    )
    return rag_chain

# Button to setup RAG
if st.sidebar.button("Siapkan Sistem"): # DIUBAH: Teks tombol lebih umum
    if openai_api_key:
        with st.spinner("Menyiapkan sistem..."):
            st.session_state['rag_chain'] = setup_rag(method_choice, model_choice, openai_api_key)
        st.sidebar.success("✅ Sistem siap!")
    else:
        st.sidebar.error("⚠️ Harap isi API key OpenAI!")

# Input Pertanyaan User
st.header("💬 Tanya AI")
user_question = st.text_input("Masukkan pertanyaan Anda di sini:")

if st.button("Dapatkan Jawaban"):
    if st.session_state['rag_chain']:
        with st.spinner("Mencari jawaban..."):
            answer = st.session_state['rag_chain'].invoke(user_question)
        st.subheader("📌 Jawaban AI:")
        st.markdown(answer)
    else:
        st.error("⚠️ Silakan siapkan sistem di sidebar terlebih dahulu!")