const sqlite3 = require('better-sqlite3'); // Substituir pelo driver mais estável
const path = require('path');

const dbPath = process.env.NODE_ENV === 'production' 
  ? '/workspace/database/questoes_medicas.db'
  : path.resolve(__dirname, '../../database/questoes_medicas.db');

// Verificar e criar diretório 
const dbDir = path.dirname(dbPath);
require('fs').mkdirSync(dbDir, { recursive: true });

const db = new sqlite3(dbPath, { 
  timeout: 5000,
  readonly: false,
  fileMustExist: false
});

// Otimizações
db.pragma('journal_mode = WAL');
db.pragma('synchronous = NORMAL');
db.pragma('cache_size = -10000');

module.exports = db;