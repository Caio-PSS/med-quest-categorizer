const sqlite3 = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

const isProduction = process.env.NODE_ENV === 'production';
const dbPath = isProduction
  ? '/workspace/database/questoes_medicas.db'
  : path.join(__dirname, '..', 'database', 'questoes_medicas.db');

// Garantir diretório existe
if (isProduction) {
  const dbDir = path.dirname(dbPath);
  if (!fs.existsSync(dbDir)) {
    fs.mkdirSync(dbDir, { recursive: true });
  }
}

// Conexão otimizada
const db = sqlite3(dbPath, {
  verbose: console.log,
  timeout: 5000
});

// Otimizações
db.pragma('journal_mode = WAL');
db.pragma('synchronous = NORMAL');
db.pragma('cache_size = -10000');

module.exports = db;