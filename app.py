# Adaptado de https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os
import base64
import gc
import tempfile
import uuid
from dotenv import load_dotenv  # NOVO: Importa a função para carregar o .env

import streamlit as st
from llama_index.core import Settings, PromptTemplate, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# NOVO: Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Inicializa o estado da sessão se ainda não existir
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

@st.cache_resource
def load_llm(model_choice, api_key=None):
    """
    Carrega o modelo de linguagem (LLM) escolhido pelo usuário.
    """
    if "Gemini" in model_choice:
        from llama_index.llms.gemini import Gemini
        llm = Gemini(model="models/gemini-1.5-flash-latest", api_key=api_key)
    else: # Padrão para Llama 3
        from llama_index.llms.ollama import Ollama
        llm = Ollama(model="llama3:8b", request_timeout=120.0)
    return llm

def reset_chat():
    """Limpa o histórico de mensagens e o contexto."""
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    """Exibe o PDF na barra lateral."""
    st.markdown("### Pré-visualização do PDF")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf" style="height:100vh; width:100%"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# --- Interface da Barra Lateral ---
with st.sidebar:
    st.header("Adicione seus documentos!")

    model_option = st.selectbox(
        "Escolha o modelo de linguagem:",
        ("Llama 3 (Local)", "Gemini-1.5-flash")
    )
    
    # ALTERADO: A chave agora é lida do ambiente, não da interface
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    uploaded_file = st.file_uploader("Escolha seu arquivo `.pdf`", type="pdf")

    if uploaded_file:
        # ALTERADO: A verificação agora é feita na variável de ambiente
        if "Gemini" in model_option and not gemini_api_key:
            st.warning("Chave da API do Gemini não encontrada. Por favor, configure seu arquivo .env.")
            st.stop()
            
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}-{model_option}"
                st.write("Indexando seu documento...")

                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                        loader = SimpleDirectoryReader(
                            input_dir=temp_dir,
                            required_exts=[".pdf"],
                            recursive=True
                        )
                    else:    
                        st.error('Não foi possível encontrar o arquivo que você enviou, por favor, verifique novamente...')
                        st.stop()
                    
                    docs = loader.load_data()
                    
                    # ALTERADO: Passa a chave obtida do .env para a função de carregamento
                    llm = load_llm(model_option, gemini_api_key)
                    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
                    
                    Settings.embed_model = embed_model
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)

                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True)

                    qa_prompt_tmpl_str = (
                        "A informação de contexto está abaixo.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Dada a informação de contexto acima, quero que você pense passo a passo para responder à pergunta de maneira clara e concisa. Caso você não saiba a resposta, diga 'Eu não sei!'.\n"
                        "Pergunta: {query_str}\n"
                        "Resposta: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )
                    
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                st.success("Pronto para conversar!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
            st.stop()    

# --- Interface Principal do Chat ---
col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Converse com seus Documentos")

with col2:
    st.button("Limpar ↺", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Qual é a sua pergunta?"):
    if not uploaded_file:
        st.warning("Por favor, faça o upload de um documento PDF primeiro.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})