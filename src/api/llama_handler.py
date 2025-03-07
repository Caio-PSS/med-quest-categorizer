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
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",  # Usará automaticamente as 2 GPUs
            attn_implementation="flash_attention_2",
            max_memory={0: "22GiB", 1: "22GiB"},  # Reserva 2GB por GPU para sistema
            offload_folder="./offload",
            rope_scaling={"type": "dynamic", "factor": 2.0}
        )
    except Exception as e:
        print(f"Erro Flash Attention: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.bfloat16
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side='left',
        model_max_length=1024,  # Redução para prevenir overflow
        truncation_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
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
    batch_size = min(
        len(questions),
        int(48 * (torch.cuda.mem_get_info(0)[0]/1e9)),  # Ajuste dinâmico pela memória
        64  # Máximo absoluto
    )
    
    # Divisão automática entre GPUs
    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15,
            num_beams=2,  # Balanceamento entre velocidade/qualidade
            do_sample=True,
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