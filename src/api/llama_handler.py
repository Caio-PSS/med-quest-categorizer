import torch  # <-- Adicione esta linha
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
import json

app = Flask(__name__)

# Configuração do Modelo
def load_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    return model, tokenizer

model, tokenizer = load_model()

@app.route('/categorize', methods=['POST'])
def categorize():
    data = request.json
    prompt = build_prompt(data['question'], data['categories'])
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify(parse_response(response))

def build_prompt(question, categories):
    return f"""
    [ENUNCIADO]: {question['enunciado']}
    [ALTERNATIVAS]: {question['alternativas']}
    [CATEGORIAS]: {categories}
    [FORMATO]: {{"categoria": "", "subtema": ""}}
    """

def parse_response(response_text):
    try:
        # Extrai o JSON da resposta usando regex
        json_str = re.search(r'\{.*\}', response_text, re.DOTALL).group()
        parsed = json.loads(json_str)
        
        # Validação básica
        if not all(key in parsed for key in ['categoria', 'subtema']):
            raise ValueError("Missing required fields")
            
        return {
            "categoria": parsed['categoria'],
            "subtema": parsed['subtema'],
            "confianca": parsed.get('confianca', 'Media'),
            "racional": parsed.get('racional', '')
        }
        
    except (AttributeError, json.JSONDecodeError, ValueError) as e:
        return {
            "categoria": "Erro",
            "subtema": str(e),
            "confianca": "Baixa",
            "racional": "Falha no parsing da resposta"
        }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)