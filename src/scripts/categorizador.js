const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const db = require('../database/database');
const axios = require('axios');

const BATCH_SIZE = 500;
const API_ENDPOINT = 'http://localhost:8888/categorize';
const API_BATCH_SIZE = 12;  // Aumentar batch size para 24
const WORKER_POOL = 6;      // Aumentar paralelismo

const { Semaphore } = require('async-mutex');
const semaphore = new Semaphore(WORKER_POOL);

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
    const rawCategories = require('../config/categorias.json');
    
    // Transformar a estrutura hierárquica em array plano de subcategorias
    const categories = Object.entries(rawCategories).flatMap(
        ([mainCat, subCats]) => subCats.map(sub => `${mainCat} - ${sub}`)
    );

    if (categories.length === 0) {
        throw new Error('Lista de categorias inválida ou vazia');
    }

    const responses = []; // ← CORREÇÃO AQUI: Inicializar o array

    for (let i = 0; i < questions.length; i += API_BATCH_SIZE) {
      const batch = questions.slice(i, i + API_BATCH_SIZE);
      try {
        const { data, status } = await axios.post(API_ENDPOINT, {
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
        }, {
          timeout: 600000,
          validateStatus: (status) => status < 500 // Aceita todos os status < 500
        });
  
        // Verificação reforçada da resposta
        if (status !== 200 || !data?.results || data.results.length !== batch.length) {
          const errorMessage = data?.error || 'Resposta inválida da API';
          console.error(`Erro no batch ${i}-${i+API_BATCH_SIZE}:`, {
            status,
            error: errorMessage,
            received: data?.results?.length || 0,
            expected: batch.length
          });
          throw new Error(`API Error: ${errorMessage}`);
        }
        
        responses.push(...data.results);
      } catch (error) {
        console.error(`Erro crítico no batch ${i}-${i+API_BATCH_SIZE}:`, {
          message: error.message,
          stack: error.stack,
          response: error.response?.data
        });
        
        // Preenche com respostas de erro detalhadas
        responses.push(...batch.map(() => ({
          categoria: 'Erro',
          subtema: error.response?.data?.error || error.message.substring(0, 100),
          confianca: 'Baixa'
        })));
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