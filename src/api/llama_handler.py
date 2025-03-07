import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json

app = Flask(__name__)

# Configuração do Modelo Mistral
def load_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

model, tokenizer = load_model()

@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data = request.json
        prompt = build_prompt(data['question'], data['categories'])
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify(parse_response(response))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def build_prompt(question, categories):
    return f"""<s>[INST] 
    Classifique esta questão médica usando apenas estas categorias:
    {json.dumps(categories, indent=2)}
    
    Dados da Questão:
    - Enunciado: {question['enunciado']}
    - Alternativas: {question['alternativas']}
    
    Formato exigido (JSON válido):
    {{
        "categoria": "Nome exato da categoria",
        "subtema": "Subcategoria correspondente"
    }}
    [/INST]</s>"""

def parse_response(response_text):
    try:
        # Extrai JSON usando delimitadores claros
        json_str = re.search(r'\{[^{}]*\}', response_text).group()
        parsed = json.loads(json_str)
        
        return {
            "categoria": parsed.get('categoria', 'Erro'),
            "subtema": parsed.get('subtema', 'Campo não encontrado'),
            "confianca": parsed.get('confianca', 'Media'),
            "raw_response": response_text  # Para debug
        }
        
    except Exception as e:
        return {
            "categoria": "Erro de análise",
            "subtema": str(e),
            "confianca": "Baixa"
        }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)  # Usando a porta 8888 conforme solicitado