const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const db = require('../database/database');
const axios = require('axios');

const BATCH_SIZE = 500;
const API_ENDPOINT = 'http://localhost:8888/categorize';

const { Semaphore } = require('async-mutex');
const semaphore = new Semaphore(4);

// Main thread logic
if (isMainThread) {
  (async () => {
    try {
      const total = await getQuestionCount();
      console.log(`Iniciando processamento de ${total} questões...`);

      const workers = [];
      for (let offset = 0; offset < total; offset += BATCH_SIZE) {
        const [_, release] = await semaphore.acquire();
        workers.push(
          new Promise((resolve) => {
            const worker = new Worker(__filename, {
              workerData: { offset, limit: BATCH_SIZE }
            });
            worker.once('message', () => {
              release();
              resolve();
            });
            worker.on('error', (error) => {
              console.error(`Erro no worker: ${error.message}`);
              release();
              resolve();
            });
          })
        );
      }

      await Promise.all(workers);
      console.log('Processamento concluído com sucesso!');
      process.exit(0);
    } catch (error) {
      console.error('Erro no processamento:', error);
      process.exit(1);
    }
  })();

  function getQuestionCount() {
    const stmt = db.prepare("SELECT COUNT(*) as total FROM perguntas");
    return stmt.get().total;
  }
} 
// Worker thread logic
else {
  (async () => {
    try {
      const { offset, limit } = workerData;
      const questions = await fetchBatch(offset, limit);
      const responses = await sendToAPI(questions);
      await updateDatabase(questions, responses);
      parentPort.postMessage(`Batch ${offset}-${offset + limit} concluído`);
    } catch (error) {
      throw new Error(`Erro no worker: ${error.message}`);
    }
  })();

  async function fetchBatch(offset, limit) {
    const stmt = db.prepare(`
      SELECT p.*, r.correta, r.explicacao 
      FROM perguntas p
      JOIN respostas r ON p.id = r.id_pergunta
      LIMIT ? OFFSET ?
    `);
    return stmt.all(limit, offset);
  }

  async function sendToAPI(questions) {
    const categories = require('../config/categorias.json');
    const responses = [];
    
    for (const question of questions) {
      try {
        const payload = {
          question: {
            enunciado: question.enunciado,
            alternativas: [
              question.alternativa_a,
              question.alternativa_b,
              question.alternativa_c || '',
              question.alternativa_d || ''
            ],
            explicacao: question.explicacao,
            correta: question.correta
          },
          categories: categories
        };

        const response = await axios.post(API_ENDPOINT, payload, {
          timeout: 30000
        });
        
        responses.push(response.data);
      } catch (error) {
        console.error(`Erro na questão ID ${question.id}:`, error.message);
        responses.push({
          categoria: 'Erro',
          subtema: 'Falha na API',
          confianca: 'Baixa'
        });
      }
    }
    
    return responses;
  }

  async function updateDatabase(questions, responses) {
    const update = db.prepare(`
      UPDATE perguntas 
      SET categoria = ?, subtema = ? 
      WHERE id = ?
    `);
  
    db.transaction(() => {
      questions.forEach((question, index) => {
        const res = responses[index];
        update.run(
          res.categoria || 'Sem categoria',
          res.subtema || 'Sem subtema',
          question.id
        );
      });
    })();
  }
}