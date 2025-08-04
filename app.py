from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_name = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.route('/')
def home():
    return "API de détection IA active."

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    texte = data.get('text', '')

    if not texte.strip():
        return jsonify({"error": "Texte vide."}), 400

    inputs = tokenizer(texte, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[0]

    return jsonify({
        "humain": round(scores[0].item() * 100, 2),
        "ia": round(scores[1].item() * 100, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
