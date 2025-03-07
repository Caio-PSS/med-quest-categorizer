const { Worker } = require('worker_threads');
const db = require('../database/database');
const axios = require('axios');

const BATCH_SIZE = 500;
const API_ENDPOINT = 'http://localhost:5000/categorize';

async function main() {
    const total = await getQuestionCount();
    
    for (let offset = 0; offset < total; offset += BATCH_SIZE) {
        new Worker('./src/scripts/worker.js', {
            workerData: { offset, limit: BATCH_SIZE }
        });
    }
}

function getQuestionCount() {
    return new Promise((resolve, reject) => {
        db.get("SELECT COUNT(*) as total FROM perguntas", (err, row) => {
            err ? reject(err) : resolve(row.total);
        });
    });
}

// Worker thread
if (!isMainThread) {
    processBatch(workerData.offset, workerData.limit);
}

async function processBatch(offset, limit) {
    const questions = await fetchBatch(offset, limit);
    const responses = await sendToAPI(questions);
    await updateDatabase(questions, responses);
}

async function fetchBatch(offset, limit) {
    return new Promise((resolve, reject) => {
      db.all(
        `SELECT p.*, r.correta, r.explicacao 
         FROM perguntas p
         JOIN respostas r ON p.id = r.id_pergunta
         LIMIT ? OFFSET ?`,
        [limit, offset],
        (err, rows) => {
          if (err) reject(err);
          else resolve(rows);
        }
      );
    });
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
        categories: require('../../config/categorias.json')
      });
      
      return response.data.results;
      
    } catch (error) {
      console.error(`API Error: ${error.message}`);
      return [];
    }
  }

main().catch(console.error);