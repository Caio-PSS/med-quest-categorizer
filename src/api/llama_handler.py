# src/api/llama_handler.py
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json

app = Flask(__name__)

# Configuração Otimizada para Mistral
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"  # Otimização para GPUs NVIDIA
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer

model, tokenizer = load_model()

@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data = request.get_json()
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
        json_str = re.search(r'\{[^{}]*\}', response_text).group()
        result = json.loads(json_str)
        
        return {
            "categoria": result.get("categoria", "Erro"),
            "subtema": result.get("subtema", "Não especificado"),
            "confianca": result.get("confianca", "Media")
        }
    except Exception as e:
        return {
            "categoria": "Erro de análise",
            "subtema": str(e),
            "confianca": "Baixa"
        }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, threaded=True)