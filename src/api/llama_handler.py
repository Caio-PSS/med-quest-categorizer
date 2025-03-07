import multiprocessing
multiprocessing.set_start_method('spawn', force=True)  # Mantido como primeira linha

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import re
import json

app = Flask(__name__)  # A variável 'app' deve existir

# Configurações de Performance
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["NCCL_P2P_DISABLE"] = "1"

# Modificação crucial para inicialização segura
model, tokenizer = None, None

def load_model():
    global model, tokenizer
    if model is None:
        torch.cuda.init()  # Inicialização explícita do CUDA
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                max_memory={0: "22GiB", 1: "22GiB"}
            )
        except Exception as e:
            print(f"Erro Flash Attention: {str(e)}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                load_in_8bit=True
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side='left',
            model_max_length=1024
        )
        
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

# Carregamento seguro do modelo
@app.before_first_request
def initialize_model():
    login(token="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")
    load_model()

    
@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data = request.get_json()
        
        if not data or 'questions' not in data or 'categories' not in data:
            return jsonify({"error": "Payload inválido"}), 400

        results = process_batch(data['questions'], data['categories'])
        return jsonify({"results": results}), 200
    
    except Exception as e:
        app.logger.error(f"Erro: {str(e)}")
        return jsonify({"error": "Erro interno"}), 500

def process_batch(questions, categories):
    prompts = [build_prompt(q, categories) for q in questions]
    
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
        add_special_tokens=True
    ).to(model.device)

    try:
        with torch.cuda.amp.autocast():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )
    except RuntimeError as e:
        return [error_response("Erro CUDA") for _ in prompts]

    return [parse_response_safe(tokenizer.decode(o, skip_special_tokens=True)) for o in outputs]


def build_prompt(question, categories):
    return f"""<s>[INST]
    ### Tarefa:
    Classifique esta questão médica usando APENAS estas categorias:

    ### Dados:
    Enunciado: {question.get('enunciado', '')}
    Alternativas: {", ".join(question.get('alternativas', []))}
    Explicação: {question.get('explicacao', '')}

    ### Categorias:
    {json.dumps(categories, indent=4)}

    ### Formato:
    {{
        "categoria": "Categoria Principal",
        "subtema": "Subcategoria",
        "confianca": "Alta/Media/Baixa"
    }}
    [/INST]</s>"""

def parse_response(text):
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if not match:
            return error_response("Formato inválido")
        
        result = json.loads(match.group())
        return {
            "categoria": result.get("categoria", "Erro"),
            "subtema": result.get("subtema", "Sem subtema"),
            "confianca": result.get("confianca", "Baixa")
        }
    except Exception as e:
        return error_response(f"Erro JSON: {str(e)}")

def error_response(message):
    return {
        "categoria": "Erro",
        "subtema": message,
        "confianca": "Baixa"
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, threaded=False, debug=False)  # Desativar threading