import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import re
import json

app = Flask(__name__)

# Configurações de Performance
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["NCCL_P2P_DISABLE"] = "1"

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Carregamento do modelo antes de iniciar a aplicação
login(token="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")

def load_model():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)  # Ativar Flash Attention

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            max_memory={0: "23GiB", 1: "23GiB"},  # Usar quase toda a VRAM
            offload_folder="./offload",
        ).eval()  # Modo avaliação para otimizações
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

# Inicialização segura
model, tokenizer = load_model()

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
    # Pré-processamento paralelizado
    with ThreadPoolExecutor(max_workers=8) as executor:
        prompts = list(executor.map(build_prompt, questions, [categories]*len(questions)))

    # Configuração de batch dinâmico
    batch_size = max(8, min(64, len(prompts)//2))  # Ajuste automático
    
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=1024,
        add_special_tokens=True,
        return_attention_mask=True
    ).to(model.device)

    # Configuração de geração otimizada
    with torch.inference_mode(), torch.cuda.amp.autocast():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15,
            do_sample=True,
            num_beams=1,           # Desativar beam search para velocidade
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decodificação assíncrona
    return [parse_response_safe(text) for text in tokenizer.batch_decode(outputs, skip_special_tokens=True)]


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
        # Extração melhorada do JSON
        text = text.split('[/INST]')[-1].strip()
        json_str = re.search(r'\{[\s\S]*\}', text).group()
        
        # Validação de campos obrigatórios
        result = json.loads(json_str)
        if not all(key in result for key in ['categoria', 'subtema', 'confianca']):
            raise ValueError("Campos obrigatórios ausentes")
            
        # Normalização dos valores
        return {
            "categoria": str(result.get("categoria", "Erro"))[:64],
            "subtema": str(result.get("subtema", "Sem subtema"))[:128],
            "confianca": str(result.get("confianca", "Baixa")).lower()[:16]
        }
    except Exception as e:
        app.logger.error(f"Erro de parsing: {str(e)} - Texto original: {text[:200]}...")
        return error_response(f"Erro de formatação: {str(e)}")

def error_response(message):
    return {
        "categoria": "Erro",
        "subtema": message,
        "confianca": "Baixa"
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, threaded=False, debug=False)