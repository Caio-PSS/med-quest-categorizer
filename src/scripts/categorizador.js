const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const db = require('../database/database');
const axios = require('axios');

const BATCH_SIZE = 500;
const API_ENDPOINT = 'http://localhost:8888/categorize';
const API_BATCH_SIZE = 4; // Ajuste conforme sua GPU

const { Semaphore } = require('async-mutex');
const semaphore = new Semaphore(4);

if (isMainThread) {
  (async () => {
    try {
      const total = await getQuestionCount();
      console.log(`Processando ${total} questões com batch de ${API_BATCH_SIZE}...`);

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
              console.error(`Worker error: ${error.message}`);
              release();
              resolve();
            });
          })
        );
      }

      await Promise.all(workers);
      console.log('Processamento concluído!');
      process.exit(0);
    } catch (error) {
      console.error('Fatal error:', error);
      process.exit(1);
    }
  })();

  function getQuestionCount() {
    return db.prepare("SELECT COUNT(*) as total FROM perguntas").get().total;
  }
} else {
  (async () => {
    try {
      const { offset, limit } = workerData;
      const questions = await fetchBatch(offset, limit);
      const responses = await sendToAPI(questions);
      await updateDatabase(questions, responses);
      parentPort.postMessage(`Batch ${offset}-${offset + limit} finalizado`);
    } catch (error) {
      throw new Error(`Worker failed: ${error.message}`);
    }
  })();

  async function fetchBatch(offset, limit) {
    return db.prepare(`
      SELECT p.*, r.correta, r.explicacao 
      FROM perguntas p
      JOIN respostas r ON p.id = r.id_pergunta
      LIMIT ? OFFSET ?
    `).all(limit, offset);
  }

  async function sendToAPI(questions) {
    const categories = require('../config/categorias.json');
    const responses = [];
    
    for (let i = 0; i < questions.length; i += API_BATCH_SIZE) {
      const batch = questions.slice(i, i + API_BATCH_SIZE);
      try {
        const payload = {
          questions: batch.map(q => ({
            enunciado: q.enunciado,
            alternativas: [
              q.alternativa_a,
              q.alternativa_b,
              q.alternativa_c || '',
              q.alternativa_d || ''
            ],
            explicacao: q.explicacao,
            correta: q.correta
          })),
          categories: categories
        };

        const { data } = await axios.post(API_ENDPOINT, payload, {
          timeout: 45000,
          headers: { 'Content-Type': 'application/json' }
        });

        responses.push(...data.results);
      } catch (error) {
        console.error(`Batch ${i}-${i+API_BATCH_SIZE} failed:`, error.message);
        batch.forEach(() => responses.push({
          categoria: 'Erro',
          subtema: 'API Error',
          confianca: 'Baixa'
        }));
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
      questions.forEach((q, index) => {
        const res = responses[index] || {};
        update.run(
          res.categoria || 'Sem categoria',
          res.subtema || 'Sem subtema',
          q.id
        );
      });
    })();
  }
}