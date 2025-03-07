const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');

const isProduction = process.env.NODE_ENV === 'production';
const dbPath = isProduction
  ? '/workspace/database/questoes_medicas.db'  // Caminho novo
  : path.join(__dirname, '..', 'database', 'questoes_medicas.db');  // Caminho local

// Garantir diretório existe em produção
if (isProduction) {
  const dbDir = path.dirname(dbPath);
  if (!fs.existsSync(dbDir)) {
    console.log(`Criando diretório persistente: ${dbDir}`);
    fs.mkdirSync(dbDir, { recursive: true });
  }
}

const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error('Database connection error:', err.message);
    process.exit(1);
  }
  console.log(`Connected to SQLite at: ${dbPath}`);
  
  // Otimizações
  db.serialize(() => {
    db.run('PRAGMA journal_mode = WAL;');
    db.run('PRAGMA synchronous = NORMAL;');
    db.run('PRAGMA cache_size = -10000;');
  });
});

module.exports = db;