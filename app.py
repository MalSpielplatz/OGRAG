import streamlit as st
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import rdflib
from collections import defaultdict

# ========== CSS BUBBLE ==========
st.set_page_config(page_title="Chatbot OG-RAG Medis", layout="wide")
st.markdown("""
    <style>
        .bubble-user {
            background-color: #e8eaf6 !important;
            color: #222 !important;
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 8px;
            margin-top: 8px;
            font-size: 1.08rem;
            width: fit-content;
            max-width: 80%;
        }
        .bubble-ai {
            background-color: #f5f5f5 !important;
            color: #16213e !important;
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 8px;
            margin-top: 8px;
            font-size: 1.08rem;
            width: fit-content;
            max-width: 80%;
        }
        .role-title {
            font-size: 0.82rem;
            color: #7a7a7a;
            margin-bottom: 5px;
            margin-top: -6px;
        }
    </style>
""", unsafe_allow_html=True)

# =================== MEMORY ==============
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Halo! Silakan tanya apa saja seputar obat, penyakit, atau medis."}
    ]

# =================== OG-RAG FUNCTIONS ===================
def uri_to_label(uri):
    uri_str = str(uri)
    return uri_str.split('#')[-1] if '#' in uri_str else uri_str.split('/')[-1]

@st.cache_resource
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

@st.cache_resource
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

# ================== LOAD MODEL & DATA ==================
RDF_PATH = "Ontology Alodog tanpa peringatan.rdf"   # Ganti ke path ontology kamu
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")        # Bisa load dari .env juga

if "faiss_index" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    st.session_state.hyperedges = construct_hypergraph_from_rdf(RDF_PATH)
    st.session_state.faiss_index = create_faiss_index(st.session_state.hyperedges, st.session_state.embeddings)

# ==================== DISPLAY HISTORY ===================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="role-title">Anda:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="role-title">AI:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble-ai">{msg["content"]}</div>', unsafe_allow_html=True)

# ================ USER INPUT ================
st.markdown("---")
with st.form(key="form_chat", clear_on_submit=True):
    user_input = st.text_input("Anda:", "", key="user_input", max_chars=400)
    submitted = st.form_submit_button("Kirim")

# =================== OG-RAG Chatbot Handler ==================
PROMPT_TEMPLATE = """
Anda adalah asisten AI medis yang akurat. Jawab pertanyaan hanya berdasarkan informasi dari "Konteks" yang disediakan.
Jika informasi tidak ada dalam konteks, jawab dengan "Informasi tidak ditemukan dalam konteks yang diberikan."

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:
"""

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})

    # --- OG-RAG Retrieval ---
    context, retrieved_docs_with_scores = retrieve_context_from_faiss_with_scores(
        user_input,
        st.session_state.faiss_index,
        st.session_state.hyperedges,
        top_k=3  # k=5, kamu bisa ubah di sini!
    )

    prompt = PROMPT_TEMPLATE.format(context=context, question=user_input)
    answer_obj = st.session_state.llm.invoke(prompt)
    answer = answer_obj.content if hasattr(answer_obj, "content") else str(answer_obj)

    st.session_state.messages.append({"role": "ai", "content": answer})
    st.experimental_rerun()

# =================== EXPORT TO CSV (optional) ===================
st.sidebar.markdown("#### Ekspor percakapan (opsional)")
if st.sidebar.button("Export ke CSV"):
    import pandas as pd
    df = pd.DataFrame(st.session_state.messages)
    df.to_csv("percakapan_chatbot.csv", index=False)
    st.sidebar.success("Percakapan berhasil diekspor ke CSV!")

