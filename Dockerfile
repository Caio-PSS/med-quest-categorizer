FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu2...

# Instala dependências
RUN apt-get update && apt-get install -y \
    curl \
    git \
    supervisor

# Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs

# Configura Supervisord
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Python
WORKDIR /app
COPY src/api/requirements.txt .
RUN pip install -r requirements.txt

# Cópia do código
COPY . .

# Permissões
RUN chmod +x /app/scripts/*.sh

VOLUME /app/persistent_storage  # Garante o mount correto

# Exposição de portas
EXPOSE 8888
EXPOSE 22

# Comando de inicialização
CMD ["bash", "-c", "npm start & python src/api/llama_handler.py --port 8888"]