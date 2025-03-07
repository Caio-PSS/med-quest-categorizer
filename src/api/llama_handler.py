import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import re
import json
import os

app = Flask(__name__)

# Configurações de Performance
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

login(token="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")

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_model():
    torch.backends.cudnn.benchmark = True
    tokenizer = None
    
    try:
        # Tentativa com Flash Attention 2
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            _attn_implementation="flash_attention_2"
        )
        print("✓ Flash Attention 2 ativado")
    except Exception as e:
        print(f"⚠️ Falha no Flash Attention 2: {str(e)}")
        try:
            # Fallback para SDPA (Scaled Dot Product Attention)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="sdpa"
            )
            print("✓ SDPA ativado como fallback")
        except Exception as e:
            print("⚠️ Fallback para implementação padrão")
            model = AutoModelForCausalrained(model_id, device_map="auto")

    # Configuração do Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side='left',
        use_fast=False,
        legacy=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    return model, tokenizer

model, tokenizer = load_model()

@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data = request.get_json()
        
        if not data or 'questions' not in data or 'categories' not in data:
            return jsonify({"error": "Payload inválido. Campos requeridos: questions, categories"}), 400

        results = process_batch(data['questions'], data['categories'])
        return jsonify({"results": results}), 200
    
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": "Erro interno"}), 500

def process_batch(questions, categories):
    prompts = [build_prompt(q, categories) for q in questions]
    
    # Tokenização do batch completo
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        add_special_tokens=True
    ).to(model.device)

    # Geração sem o parâmetro batch_size
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True
    )
    
    return [parse_response(tokenizer.decode(o, skip_special_tokens=True)) for o in outputs]

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
    app.run(host='0.0.0.0', port=8888, threaded=True, debug=False)