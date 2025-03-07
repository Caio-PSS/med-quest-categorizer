const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const db = require('../database/database');
const axios = require('axios');

const BATCH_SIZE = 500;
const API_ENDPOINT = 'http://localhost:8888/categorize'; // Porta corrigida para 8888

const { Semaphore } = require('async-mutex');
const semaphore = new Semaphore(4); // Limitar concorrência

// Main thread logic
if (isMainThread) {
  (async () => {
    try {
      const total = await getQuestionCount();
      console.log(`Iniciando processamento de ${total} questões...`);

      const workers = [];
      for (let offset = 0; offset < total; offset += BATCH_SIZE) {
        const release = await semaphore.acquire();
        workers.push(
          new Promise((resolve) => {
            const worker = new Worker(__filename, {
              workerData: { offset, limit: BATCH_SIZE }
            });
            worker.on('message', () => {
              release();
              resolve();
            });
          })
        );
      }

      await Promise.all(workers);
      console.log('Processamento concluído com sucesso!');
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
    try {
      const response = await axios.post(API_ENDPOINT, {
        questions: questions.map(q => ({
          enunciado: q.enunciado,
          alternativas: [
            q.alternativa_a,
            q.alternativa_b,
            q.alternativa_c || '',
            q.alternativa_d || ''
          ],
          categoria: q.categoria,
          correta: q.correta,
          explicacao: q.explicacao
        })),
        categories: require('../config/categorias.json')
      });
      
      return response.data.results;
    } catch (error) {
      console.error(`API Error: ${error.message}`);
      return [];
    }
  }

  async function updateDatabase(questions, responses) {
    const update = db.prepare(`
      UPDATE perguntas 
      SET categoria = ?, subtema = ? 
      WHERE id = ?
    `);
  
    db.transaction(() => {
      for (const [index, res] of responses.entries()) {
        update.run(res.categoria, res.subtema, questions[index].id);
      }
    })();
  }
}