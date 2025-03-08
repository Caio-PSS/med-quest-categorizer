import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import re
import json
import traceback
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Configurações de Performance
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["NCCL_P2P_DISABLE"] = "1"

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_model():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            max_memory={0: "22GiB", 1: "22GiB"},
            offload_folder="./offload",
        ).eval()
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
    
    return model, tokenizer

# Inicialização do modelo
login(token="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")
model, tokenizer = load_model()

@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data = request.get_json()
        
        # Validação aprimorada
        if not data or not isinstance(data.get('categories'), list):
            return jsonify({"error": "Formato de categorias inválido"}), 400
            
        valid_categories = [
            c for c in data['categories']
            if isinstance(c, str) and ' - ' in c
        ]
        
        if len(valid_categories) == 0:
            return jsonify({"error": "Categorias devem seguir formato 'Categoria - Subcategoria'"}), 400
            
        if not isinstance(data.get('questions'), list) or len(data['questions']) == 0:
            return jsonify({"error": "Lista de questões inválida"}), 400
            
        if len(data['questions']) > 100:
            return jsonify({"error": "Máximo de 100 questões por requisição"}), 413
            
        # Filtra questões inválidas
        valid_questions = [
            q for q in data['questions']
            if isinstance(q, dict) and 'enunciado' in q and len(q['enunciado']) > 10
        ]
        
        if not valid_questions:
            return jsonify({"error": "Nenhuma questão válida encontrada"}), 400

        results = process_batch(valid_questions, valid_categories)
        return jsonify({"results": results}), 200
    
    except Exception as e:
        app.logger.error(f"ERRO GLOBAL: {traceback.format_exc()}")
        return jsonify({"error": "Falha no processamento"}), 500

def process_batch(questions, categories):
    with ThreadPoolExecutor(max_workers=8) as executor:
        prompts = list(executor.map(build_prompt, questions, [categories]*len(questions)))

    batch_size = max(4, min(32, len(prompts)//2))
    
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=1024,
        add_special_tokens=True,
        return_attention_mask=True
    ).to(model.device)

    try:
        with torch.inference_mode(), torch.amp.autocast(device_type='cuda'):
            outputs = model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=[[tokenizer.encode('\n')[0]]],
                max_time=300
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return [error_response("Memória insuficiente")] * len(questions)

    return [parse_response(text) for text in tokenizer.batch_decode(outputs, skip_special_tokens=True)]

def build_prompt(question, categories):
    def clean_text(text, max_length):
        return text.replace('"', "'").replace('\n', ' ').strip()[:max_length]
    
    return f"""<s>[INST]
    ### FORMATO EXATO REQUERIDO:
    {{
        "categoria": "Categoria Principal",
        "subtema": "Subcategoria Específica",
        "confianca": "Alta/Media/Baixa"
    }}

    ### DADOS DA QUESTÃO:
    Enunciado: {clean_text(question.get('enunciado', ''), 800)}
    Alternativas: {json.dumps([clean_text(a, 100) for a in question.get('alternativas', [])])[:400]}
    Explicação: {clean_text(question.get('explicacao', ''), 500)}

    ### CATEGORIAS VÁLIDAS:
    {', '.join(sorted(set(categories)))[:1200]}
    [/INST]"""

def parse_response(text):
    sanitized = ""
    
    def repair_json(raw_json):
        repairs = [
            (r'(?<!\\)\\(?!["\\/bfnrt])', ''),
            (r',\s*}(?=\s*})', '}'),
            (r'\bNaN\b', 'null'),
            (r'\s+', ' '),
            (r'([{\[,])\s*([}\]])', r'\1\2'),
            (r"'", '"')
        ]
        
        for pattern, replacement in repairs:
            raw_json = re.sub(pattern, replacement, raw_json)
            
        open_braces = raw_json.count('{') - raw_json.count('}')
        if open_braces > 0:
            raw_json += '}' * open_braces
            
        return raw_json.strip('`').strip()

    try:
        # Pré-processamento
        text = re.sub(r'```json|```|\*+', '', text)
        text = text.replace('\\"', "'")
        
        # Extração de JSON
        json_match = re.search(r'\{[\s\S]*\}', text, re.DOTALL)
        
        if not json_match:
            raise ValueError("Nenhuma estrutura JSON detectada")
            
        raw_json = json_match.group(0)
        sanitized = repair_json(raw_json)

        # Validação
        result = json.loads(sanitized)
        
        if not all(key in result for key in ['categoria', 'subtema', 'confianca']):
            raise ValueError("Campos obrigatórios ausentes")
            
        # Normalização
        confianca = str(result['confianca']).lower().strip()
        if confianca not in ['alta', 'media', 'baixa']:
            confianca = 'baixa'
            
        return {
            'categoria': str(result['categoria']).strip()[:64],
            'subtema': str(result['subtema']).strip()[:128],
            'confianca': confianca[:5]
        }
        
    except Exception as e:
        error_type = type(e).__name__
        fallback_response = None
        
        try:
            last_part = re.search(r'\{.*', text, re.DOTALL)
            if last_part:
                sanitized = repair_json(last_part.group(0))
                result = json.loads(sanitized)
                fallback_response = {
                    'categoria': str(result.get('categoria', 'Erro')).strip()[:64],
                    'subtema': str(result.get('subtema', 'Parseamento parcial')).strip()[:128],
                    'confianca': 'baixa'
                }
        except:
            pass

        app.logger.error(f"FALHA PARSING [{error_type}]: {str(e)}\n"
                        f"Texto original: {text[:300]}\n"
                        f"JSON sanitizado: {sanitized[:500] if sanitized else 'N/A'}\n"
                        f"{'─'*80}")
        
        return fallback_response or error_response(f"Erro {error_type}: {str(e)[:80]}")

def error_response(message):
    return {
        "categoria": "Erro",
        "subtema": str(message)[:100],
        "confianca": "baixa"
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, threaded=False, debug=False)