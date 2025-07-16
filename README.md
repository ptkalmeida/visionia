# Vision IA 🧠📄

Uma inteligência artificial que lê, entende e responde com base em documentos como PDFs, Word (.docx), textos (.txt) e planilhas (.xlsx).

> Projeto em desenvolvimento para modernizar o atendimento e a análise de documentos com inteligência artificial de verdade (sem alucinação 🎯).

---

## 🔍 O que ela faz

- 📂 Ingestão automática de arquivos
- 🧠 Geração de vetores com FAISS + BAAI/bge-m3
- 🗂️ Armazena somente documentos novos (com controle incremental)
- 🔎 Busca inteligente por contexto real (sem consulta inventada)
- 📞 Consulta por ramais a partir de um Excel com fuzzy match
- 🌐 Interface web para perguntas diretas
- 🧾 Respostas baseadas nos documentos, com geração por IA

---

## 🚀 Como rodar localmente

1. Clone o repositório:

```bash
git clone https://github.com/ptkalmeida/visionia.git
cd visionia

2. Crie o ambiente virtual:

bash
Copiar
Editar
python -m venv .venv
.venv\\Scripts\\activate

3. Instale as dependências:

bash
Copiar
Editar
pip install -r requirements.txt

4. Rode o app:

bash
Copiar
Editar
python app.py
Acesse em http://localhost:5000

