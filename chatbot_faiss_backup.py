# iaedit_pipeline_upgrade - VERSÃO OTIMIZADA

import os
import re
import json
import logging
import requests
import pdfplumber
import pandas as pd
from unidecode import unidecode
from fuzzywuzzy import fuzz
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # CORRIGIDO
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

# CONFIGURAÇÕES OTIMIZADAS
PASTA_DOCUMENTOS = "data/editais"
CHROMA_PATH = "db"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "openchat"
CHUNK_SIZE = 600  # Reduzido de 800
CHUNK_OVERLAP = 150  # Reduzido de 200

# LOGS
logging.basicConfig(filename='chatbot.log', level=logging.INFO)

# EMBEDDING E FAISS (ATUALIZADO)
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
db = FAISS.load_local(CHROMA_PATH, embeddings=embedding, allow_dangerous_deserialization=True)

# CACHE GLOBAL
DOCUMENTOS_CACHE = None
CACHE_RESPOSTAS = {}  # NOVO: Cache de respostas

# FUNÇÕES DE SUPORTE
def normalize_text(text):
    text = unidecode(text.lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_modelo(texto):
    markers = ["[Nome do projeto]", "[dd/mm/2025]", "Ex:", "Exemplo:", "preencher", "modelo", "template"]
    return any(m in texto for m in markers)

def extrair_estrutura_modelo(texto):
    secoes = re.findall(r"(?<=\n)[A-Z ]{3,}(?=\n)", texto)
    return list(set(secoes))

def carregar_documentos():
    documentos = []
    for nome_arquivo in os.listdir(PASTA_DOCUMENTOS):
        caminho = os.path.join(PASTA_DOCUMENTOS, nome_arquivo)
        try:
            if nome_arquivo.endswith(".pdf"):
                with pdfplumber.open(caminho) as pdf:
                    for page in pdf.pages:
                        for table in page.extract_tables():
                            for row in table:
                                if len(row) >= 2:
                                    content = " | ".join(str(cell) for cell in row if cell)
                                    documentos.append(Document(
                                        page_content=f"TABELA: {content}",
                                        metadata={"source": nome_arquivo, "pagina": page.page_number, "tipo": "tabela"}
                                    ))
                        text = page.extract_text()
                        if text:
                            metadata = {"source": nome_arquivo, "pagina": page.page_number, "tipo": "real"}
                            if is_modelo(text):
                                metadata["tipo"] = "modelo"
                                metadata["estrutura"] = extrair_estrutura_modelo(text)
                            documentos.append(Document(
                                page_content=text,
                                metadata=metadata
                            ))
            elif nome_arquivo.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(caminho)
                documentos.extend(loader.load())
            elif nome_arquivo.endswith(".txt"):
                loader = TextLoader(caminho, encoding="utf-8")
                documentos.extend(loader.load())
            elif nome_arquivo.endswith(".json"):
                loader = JSONLoader(
                    file_path=caminho,
                    jq_schema=".",
                    text_content=False,
                    metadata_func=lambda _: {"source": nome_arquivo},
                )
                documentos.extend(loader.load())
        except pdfplumber.pdf.PDFSyntaxError:
            logging.warning(f"PDF corrompido: {nome_arquivo}")
        except Exception as e:
            logging.error(f"Erro ao carregar {nome_arquivo}: {str(e)}")
    return documentos

def get_documentos_cache():
    global DOCUMENTOS_CACHE
    if DOCUMENTOS_CACHE is None:
        DOCUMENTOS_CACHE = carregar_documentos()
    return DOCUMENTOS_CACHE

def gerar_chunks(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "(?<=\\. )", " ", ""]
    )
    return splitter.split_documents(documentos)

def salvar_em_faiss(chunks):
    db = FAISS.from_documents(chunks, embedding)
    db.save_local(CHROMA_PATH)

def contem_nome_flex(doc, nome):
    return unidecode(nome.lower()) in unidecode(doc.page_content.lower())

def buscar_keywords(pergunta, documentos, top_n=5):  # Reduzido de 10 para 5
    vectorizer = TfidfVectorizer(stop_words=None)
    tfidf_matrix = vectorizer.fit_transform([doc.page_content for doc in documentos])
    query_vec = vectorizer.transform([pergunta])
    scores = (tfidf_matrix * query_vec.T).toarray().flatten()
    return [documentos[i] for i in scores.argsort()[-top_n:]]

def buscar_contexto(pergunta, nome=None, k=15):  # OTIMIZADO: k reduzido de 30 para 15
    try:
        resultados_faiss = db.max_marginal_relevance_search(
            pergunta, 
            k=k, 
            fetch_k=50  # Reduzido de 100 para 50
        )
    except Exception as e:
        logging.error(f"Erro no FAISS: {str(e)}")
        resultados_faiss = db.similarity_search(pergunta, k=k)

    documentos = get_documentos_cache()
    resultados_keywords = buscar_keywords(pergunta, documentos, top_n=5)
    resultados_combinados = resultados_faiss + resultados_keywords

    vistos = set()
    final = []
    for doc in resultados_combinados:
        text = doc.page_content.strip()
        if text not in vistos:
            vistos.add(text)
            final.append(doc)

    if nome:
        nome = unidecode(nome.lower())
        final = [doc for doc in final if contem_nome_flex(doc, nome)]

    if "modelo" in pergunta.lower() or "formato" in pergunta.lower():
        final = [doc for doc in final if doc.metadata.get("tipo") == "modelo"]
    elif "dados" in pergunta.lower() or "informacao real" in pergunta.lower():
        final = [doc for doc in final if doc.metadata.get("tipo") != "modelo"]

    if not final:
        return ""

    # OTIMIZADO: Limitar a 8 documentos e 2500 chars por contexto
    contexto = "\n\n".join(
        [f"[{doc.metadata.get('source', 'Desconhecido')}, p. {doc.metadata.get('pagina', '?')}]:\n{doc.page_content[:400]}" 
         for doc in final[:8]]
    )
    return contexto[:2500]  # Limitar contexto total

def perguntar_para_ia_otimizada(pergunta, contexto, modo="detalhado"):
    """VERSÃO SUPER OTIMIZADA com cache"""
    
    # Verificar cache primeiro
    cache_key = f"{pergunta[:50]}_{hash(contexto[:500])}"
    if cache_key in CACHE_RESPOSTAS:
        return f"[CACHE] {CACHE_RESPOSTAS[cache_key]}"
    
    # Template mais enxuto
    template = f"""CONTEXTO:
{contexto}

PERGUNTA: {pergunta}

RESPOSTA OBJETIVA:"""

    payload = {
        "model": MODEL,
        "prompt": template,
        "stream": False,
        "options": {
            "temperature": 0.3,      # Criatividade baixa = mais rápido
            "top_k": 15,            # Poucas opções = mais rápido
            "top_p": 0.8,           # Menos variabilidade = mais rápido
            "num_ctx": 1024,        # MUITO IMPORTANTE: Contexto pequeno = MUITO mais rápido
            "num_predict": 300,     # Respostas limitadas = mais rápido
            "repeat_penalty": 1.1,  # Evita repetições
            "num_thread": 4         # Usar múltiplos threads
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL, 
            json=payload, 
            timeout=120  # Reduzido de 800 para 120 segundos
        )
        resposta = response.json()["response"]
        
        # Salvar no cache (limitado a 50 respostas)
        if len(CACHE_RESPOSTAS) < 50:
            CACHE_RESPOSTAS[cache_key] = resposta
        
        return resposta
    except requests.exceptions.Timeout:
        return "⏱️ Timeout: Resposta demorou muito. Tente uma pergunta mais específica."
    except Exception as e:
        logging.error(f"Erro ao consultar modelo: {str(e)}")
        return "❌ Erro ao gerar resposta."

# FUNÇÃO ORIGINAL (para compatibilidade)
def perguntar_para_ia(pergunta, contexto, modo="detalhado"):
    return perguntar_para_ia_otimizada(pergunta, contexto, modo)

def validar_resposta(resposta, contexto):
    return any(fuzz.partial_ratio(resposta, ctx) > 85 for ctx in contexto.split("\n\n"))

# FUNÇÕES DE DIAGNÓSTICO
def diagnosticar_performance():
    """Diagnóstica possíveis problemas de performance"""
    print("🔍 DIAGNÓSTICO DE PERFORMANCE:")
    
    # Verificar modelos disponíveis
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        modelos = response.json()['models']
        print(f"📊 Modelos disponíveis: {len(modelos)}")
        
        # Sugerir modelo mais leve
        modelos_rapidos = ["openchat:3.5-3b", "openchat:3.5-7b-q4_0"]
        for modelo in modelos_rapidos:
            if any(modelo in m['name'] for m in modelos):
                print(f"⚡ RECOMENDAÇÃO: Use '{modelo}' para mais velocidade")
                break
    except:
        print("❌ Erro ao conectar com Ollama")
    
    # Verificar cache
    print(f"💾 Cache atual: {len(CACHE_RESPOSTAS)} respostas")
    
    # Verificar documentos
    docs = get_documentos_cache()
    print(f"📄 Documentos carregados: {len(docs)}")

def limpar_cache():
    """Limpa o cache de respostas"""
    global CACHE_RESPOSTAS
    CACHE_RESPOSTAS = {}
    print("🧹 Cache limpo!")

if __name__ == "__main__":
    print("🚀 Iniciando diagnóstico...")
    diagnosticar_performance()
    
    # Exemplo de uso otimizado
    print("\n📝 Teste rápido:")
    pergunta = "O que é este sistema?"
    contexto = "Sistema de chatbot com IA"
    resposta = perguntar_para_ia_otimizada(pergunta, contexto)
    print(f"Resposta: {resposta}")