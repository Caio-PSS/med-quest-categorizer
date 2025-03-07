import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import re
import json

app = Flask(__name__)

# Configuração segura
login(token="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")

# Configuração do Modelo
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

model, tokenizer = load_model()

@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data = request.get_json()
        
        # Validação reforçada
        if not data or 'question' not in data or 'categories' not in data:
            return jsonify({
                "error": "Payload inválido. Campos obrigatórios: 'question', 'categories'"
            }), 400

        # Construção do prompt
        prompt = build_prompt(data['question'], data['categories'])
        
        # Geração da resposta
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True
        )
        
        # Processamento da resposta
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        parsed = parse_response(response)
        
        return jsonify(parsed), 200
    
    except Exception as e:
        app.logger.error(f"Erro interno: {str(e)}")
        return jsonify({"error": "Erro interno no processamento"}), 500

def build_prompt(question, categories):
    return f"""<s>[INST] 
    ### Tarefa:
    Classifique esta questão médica usando EXCLUSIVAMENTE as categorias fornecidas.

    ### Dados da Questão:
    Enunciado: {question.get('enunciado', '')}
    Alternativas: {", ".join(question.get('alternativas', []))}
    Explicação Atual: {question.get('explicacao', '')}

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
        json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "categoria": result.get("categoria", "Erro"),
                "subtema": result.get("subtema", "Sem subtema"),
                "confianca": result.get("confianca", "Baixa")
            }
        return {"categoria": "Erro", "subtema": "Formato inválido", "confianca": "Baixa"}
    except Exception as e:
        return {"categoria": "Erro", "subtema": f"Erro de parsing: {str(e)}", "confianca": "Baixa"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, threaded=True, debug=False)