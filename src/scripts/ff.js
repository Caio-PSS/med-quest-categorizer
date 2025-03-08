const sqlite3 = require('better-sqlite3');
const db = new sqlite3('/workspace/database/questoes_medicas.db');

// Verificar se a tabela 'perguntas' existe
const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table';").all();
console.log(tables);
