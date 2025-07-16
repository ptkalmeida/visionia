from flask import Flask, request, render_template
from chatbot_faiss import get_ai_response

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    resposta = ""
    if request.method == "POST":
        pergunta = request.form.get("pergunta")
        resposta = get_ai_response(pergunta)
    return render_template("index.html", resposta=resposta)

if __name__ == "__main__":
    app.run(debug=True)
