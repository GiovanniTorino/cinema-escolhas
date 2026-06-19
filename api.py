from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

GENEROS = {
    0: "Ação",
    1: "Comédia",
    2: "Drama",
    3: "Terror",
    4: "Romance",
    5: "Animação",
}
GENERO_PARA_COD = {v: k for k, v in GENEROS.items()}

FILMES_RAW = [
    ("Velozes e Furiosos",       0, 120, 7.8, 2019, 85),
    ("Dunkirk",                  0, 150, 6.5, 2021, 60),
    ("Hawaii 5-0",               0, 130, 8.2, 2023, 95),
    ("Invasão à Casa Branca",    0, 110, 5.9, 2018, 40),

    ("Piada Mortal",           1,  95, 7.0, 2020, 70),
    ("Family Guy",             1, 100, 6.2, 2017, 50),
    ("Projeto X",              1,  90, 8.0, 2022, 88),
    ("O Pestinha",             1, 105, 5.5, 2016, 35),

    ("Marley e Eu",             2, 140, 8.5, 2021, 75),
    ("Resgate do Soldado Ryan", 2, 125, 7.2, 2019, 55),
    ("A Culpa É das Estrelas",  2, 135, 9.0, 2023, 90),
    ("O Ultimo Homem",          2, 115, 6.0, 2015, 30),

    ("Até a morte",            3,  90, 6.8, 2020, 65),
    ("A Cabana",               3, 100, 7.5, 2022, 80),
    ("Panico 5",               3,  85, 5.0, 2014, 25),
    ("Eu sei oque vc fez verão passado",          3,  95, 8.1, 2023, 92),

    ("Amor & Outras Drogas",   4, 105, 7.6, 2021, 78),
    ("Amor à Queima-Roupa",    4, 110, 8.3, 2022, 84),
    ("Spring Breakers",        4,  98, 6.4, 2018, 45),
    ("Obsessão",               4, 102, 5.8, 2016, 38),

    ("Hobbit",                 5,  90, 8.7, 2023, 96),
    ("Magico de Oz",           5,  95, 7.4, 2020, 70),
    ("Pequenos Espiões",       5,  88, 6.1, 2017, 42),
    ("Peter Pan",              5,  92, 9.2, 2023, 98),
]

df = pd.DataFrame(
    FILMES_RAW,
    columns=["titulo", "genero_cod", "duracao_min", "avaliacao", "ano", "popularidade"],
)
df["genero"] = df["genero_cod"].map(GENEROS)


df["recomendado"] = (df["avaliacao"] >= 7.5).astype(int)

FEATURE_COLS = ["genero_cod", "duracao_min", "avaliacao", "ano", "popularidade"]

X = df[FEATURE_COLS].values   
y = df["recomendado"].values  


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
modelo.fit(X_train, y_train)

y_pred_teste = modelo.predict(X_test)
ACURACIA = accuracy_score(y_test, y_pred_teste)
print(f"[INFO] Modelo treinado. Acurácia no conjunto de teste: {ACURACIA:.2f}")

def calcular_score(linha):
    features = np.array([[ 
        linha["genero_cod"],
        linha["duracao_min"],
        linha["avaliacao"],
        linha["ano"],
        linha["popularidade"],
    ]])
    proba = modelo.predict_proba(features)[0]
    return round(float(proba[1]), 3)


df["score_recomendacao"] = df.apply(calcular_score, axis=1)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "mensagem": "API de Recomendação de Filmes (Aprendizado Supervisionado)",
        "modelo": "DecisionTreeClassifier",
        "acuracia_teste": round(ACURACIA, 3),
        "endpoints": {
            "GET /generos": "lista os gêneros disponíveis",
            "GET /filmes?genero=Comédia": "lista filmes (filtro opcional por gênero)",
            "GET /matriz": "mostra a matriz X e o vetor y usados no treinamento",
            "POST /recomendar": "recomenda filme(s) por gênero + critério (duracao ou avaliacao)",
        },
    })


@app.route("/generos", methods=["GET"])
def listar_generos():
    return jsonify({"generos": list(GENEROS.values())})


@app.route("/filmes", methods=["GET"])
def listar_filmes():
    genero = request.args.get("genero")
    resultado = df
    if genero:
        resultado = resultado[resultado["genero"].str.lower() == genero.lower()]
        if resultado.empty:
            return jsonify({"erro": f"Nenhum filme encontrado para o gênero '{genero}'."}), 404

    colunas = ["titulo", "genero", "duracao_min", "avaliacao", "ano",
               "popularidade", "score_recomendacao"]
    return jsonify(resultado[colunas].to_dict(orient="records"))


@app.route("/matriz", methods=["GET"])
def mostrar_matriz():
    """Endpoint didático: expõe a matriz X e o vetor y do trabalho."""
    return jsonify({
        "colunas_X": FEATURE_COLS,
        "X": X.tolist(),
        "y": y.tolist(),
    })


@app.route("/recomendar", methods=["POST"])
def recomendar():
    dados = request.get_json(silent=True) or {}

    genero = dados.get("genero")
    criterio = dados.get("criterio")  # "duracao" ou "avaliacao"

    if not genero or genero not in GENERO_PARA_COD:
        return jsonify({
            "erro": "Informe um 'genero' válido.",
            "generos_validos": list(GENEROS.values()),
        }), 400

    if criterio not in ("duracao", "avaliacao"):
        return jsonify({
            "erro": "Informe 'criterio' como 'duracao' ou 'avaliacao'."
        }), 400

    candidatos = df[df["genero"].str.lower() == genero.lower()].copy()

    if criterio == "duracao":
        duracao_max = dados.get("duracao_max")
        if duracao_max is None:
            return jsonify({"erro": "Informe 'duracao_max' (em minutos) quando criterio='duracao'."}), 400
        candidatos = candidatos[candidatos["duracao_min"] <= float(duracao_max)]
        ordenacao = ["score_recomendacao", "duracao_min"]
        ascendente = [False, True]
    else: 
        avaliacao_min = dados.get("avaliacao_min")
        if avaliacao_min is None:
            return jsonify({"erro": "Informe 'avaliacao_min' quando criterio='avaliacao'."}), 400
        candidatos = candidatos[candidatos["avaliacao"] >= float(avaliacao_min)]
        ordenacao = ["score_recomendacao", "avaliacao"]
        ascendente = [False, False]

    if candidatos.empty:
        return jsonify({
            "mensagem": "Nenhum filme encontrado com esses critérios. "
                        "Tente afrouxar o filtro de duração/avaliação."
        }), 200

    candidatos = candidatos.sort_values(by=ordenacao, ascending=ascendente)

    colunas = ["titulo", "genero", "duracao_min", "avaliacao", "ano",
               "popularidade", "score_recomendacao"]

    top3 = candidatos.head(3)[colunas].to_dict(orient="records")

    return jsonify({
        "genero_consultado": genero,
        "criterio_usado": criterio,
        "total_candidatos": int(len(candidatos)),
        "recomendacoes": top3,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
