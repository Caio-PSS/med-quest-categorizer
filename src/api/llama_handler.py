import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import re
import json

app = Flask(__name__)

# 🔒 Substitua pelo seu token seguro
login(token="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")

# 🔥 Configuração do Modelo otimizada para múltiplas GPUs
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="balanced_low_0"  # 🔥 Distribui entre as GPUs automaticamente
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

model, tokenizer = load_model()

@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data = request.get_json()
        
        if not data or 'question' not in data or 'categories' not in data:
            return jsonify({"error": "Requisição inválida. Certifique-se de incluir 'question' e 'categories'."}), 400

        prompt = build_prompt(data['question'], data['categories'])
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify(parse_response(response)), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def build_prompt(question, categories):
    return f"""<s>[INST] 
    ### Tarefa:
    Classifique esta questão médica usando EXCLUSIVAMENTE as categorias fornecidas.

    ### Dados da Questão:
    Enunciado: {question['enunciado']}
    Alternativas: {", ".join(question['alternativas'])}
    Explicação Atual: {question['explicacao']}

    ### Categorias Disponíveis:
    {json.dumps(categories, indent=4)}

    ### Formato de Resposta (JSON):
    {{
        "categoria": "Categoria Principal",
        "subtema": "Subcategoria Específica",
        "confianca": "Alta/Media/Baixa"
    }}
    [/INST]</s>"""

def parse_response(response_text):
    try:
        json_str = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if json_str:
            return json.loads(json_str.group())
        else:
            return {"categoria": "Erro", "subtema": "Nenhum JSON encontrado", "confianca": "Baixa"}
    except json.JSONDecodeError as e:
        return {"categoria": "Erro", "subtema": f"Erro JSON: {str(e)}", "confianca": "Baixa"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, threaded=True)
