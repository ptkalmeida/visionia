import os
import fitz  # PyMuPDF
import docx
import json
from pathlib import Path
from unidecode import unidecode

def extrair_texto_pdf(caminho_pdf):
    try:
        doc = fitz.open(caminho_pdf)
        texto = ""
        for pagina in doc:
            texto += pagina.get_text()
        return texto
    except Exception as e:
        print(f"[ERRO] Erro ao extrair texto de {caminho_pdf}: {e}")
        return ""

def salvar_txt(texto, caminho_txt):
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write(texto)

def salvar_docx(texto, caminho_docx):
    doc = docx.Document()
    for linha in texto.split("\n"):
        doc.add_paragraph(linha)
    doc.save(caminho_docx)

def extrair_dados_estruturados(texto):
    dados = []
    blocos = texto.split("\n\n")
    for bloco in blocos:
        entrada = {}
        for linha in bloco.split("\n"):
            if ":" in linha:
                chave, valor = linha.split(":", 1)
                entrada[unidecode(chave.strip().lower().replace(" ", "_"))] = valor.strip()
        if entrada:
            dados.append(entrada)
    return dados

def salvar_json(dados, caminho_json):
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

def processar_arquivos_pdf(pasta_pdf, pasta_saida):
    os.makedirs(pasta_saida, exist_ok=True)
    for arquivo in os.listdir(pasta_pdf):
        if arquivo.endswith(".pdf"):
            caminho_pdf = os.path.join(pasta_pdf, arquivo)
            nome_base = Path(arquivo).stem
            texto = extrair_texto_pdf(caminho_pdf)

            if texto:
                salvar_txt(texto, os.path.join(pasta_saida, f"{nome_base}.txt"))
                salvar_docx(texto, os.path.join(pasta_saida, f"{nome_base}.docx"))
                dados = extrair_dados_estruturados(texto)
                salvar_json(dados, os.path.join(pasta_saida, f"{nome_base}.json"))
                print(f"[OK] Processado: {arquivo}")
            else:
                print(f"[AVISO] Nenhum texto extraído de {arquivo}")
