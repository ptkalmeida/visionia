import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    JSONLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Caminhos
PASTA_DOCUMENTOS = "data/editais"
VETOR_PATH = "db"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150

# Modelo de embedding BGE-M3
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

def carregar_documentos():
    documentos = []

    for nome_arquivo in os.listdir(PASTA_DOCUMENTOS):
        caminho = os.path.join(PASTA_DOCUMENTOS, nome_arquivo)

        try:
            if nome_arquivo.endswith(".pdf"):
                import pdfplumber
                with pdfplumber.open(caminho) as pdf:
                    text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                documentos.append(Document(page_content=text, metadata={"source": nome_arquivo}))
                print(f"📄 Documento carregado: {nome_arquivo}")
            elif nome_arquivo.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(caminho)
                documentos.extend(loader.load())
                print(f"📄 Documento carregado: {nome_arquivo}")
            elif nome_arquivo.endswith(".txt"):
                loader = TextLoader(caminho, encoding="utf-8")
                documentos.extend(loader.load())
                print(f"📄 Documento carregado: {nome_arquivo}")
            elif nome_arquivo.endswith(".json"):
                loader = JSONLoader(
                    file_path=caminho,
                    jq_schema=".",
                    text_content=False,
                    metadata_func=lambda _: {"source": nome_arquivo},
                )
                documentos.extend(loader.load())
                print(f"📄 Documento carregado: {nome_arquivo}")
            else:
                print(f"❌ Tipo de arquivo não suportado: {nome_arquivo}")
        except Exception as e:
            print(f"⚠️ Erro ao carregar {nome_arquivo}: {e}")

    return documentos

def gerar_chunks(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n## ", "\n\n", "\n• ", "\n", " "]
    )
    return splitter.split_documents(documentos)

def salvar_em_faiss(chunks):
    print("🔎 Gerando embeddings e salvando no FAISS...")
    db = FAISS.from_documents(chunks, embedding)
    db.save_local(VETOR_PATH)
    print("✅ Vetorização concluída com sucesso.")

if __name__ == "__main__":
    print("🚀 Iniciando processo de ingestão de documentos...")
    docs = carregar_documentos()

    if not docs:
        print("❌ Nenhum documento válido encontrado.")
        exit()

    docs_chunked = gerar_chunks(docs)
    salvar_em_faiss(docs_chunked)
