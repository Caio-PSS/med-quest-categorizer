const db = require('../database');

// Schema completo conforme especificado
db.serialize(() => {
  db.exec(`
    CREATE TABLE IF NOT EXISTS perguntas (
      id TEXT PRIMARY KEY,
      enunciado TEXT NOT NULL,
      alternativa_a TEXT NOT NULL,
      alternativa_b TEXT NOT NULL,
      alternativa_c TEXT,
      alternativa_d TEXT,
      categoria TEXT,
      subtema TEXT
    );

    CREATE TABLE IF NOT EXISTS respostas (
      id_pergunta TEXT PRIMARY KEY,
      correta TEXT NOT NULL,
      explicacao TEXT NOT NULL,
      FOREIGN KEY(id_pergunta) REFERENCES perguntas(id)
    );

    CREATE TABLE IF NOT EXISTS historico_respostas (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      id_questao TEXT NOT NULL,
      resposta_usuario TEXT NOT NULL,
      resposta_correta TEXT NOT NULL,
      resultado TEXT NOT NULL,
      data_resposta TEXT NOT NULL,
      tempo_resposta REAL NOT NULL,
      FOREIGN KEY(id_questao) REFERENCES perguntas(id)
    );

    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      email TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS challenges (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      nome TEXT NOT NULL,
      descricao TEXT,
      tipo TEXT NOT NULL CHECK(tipo IN ('desempenho', 'quantidade')),
      meta TEXT,
      data_inicio TEXT NOT NULL,
      data_fim TEXT NOT NULL,
      CHECK(json_valid(meta) OR meta IS NULL)
    );

    CREATE TABLE IF NOT EXISTS user_challenges (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      challenge_id INTEGER NOT NULL,
      status TEXT DEFAULT 'pendente' CHECK(status IN ('pendente', 'concluido')),
      data_inicio TEXT DEFAULT CURRENT_TIMESTAMP,
      data_conclusao TEXT,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
      FOREIGN KEY(challenge_id) REFERENCES challenges(id) ON DELETE CASCADE
    );
  `, (err) => {
    if (err) console.error('Migration error:', err);
    else console.log('All tables created successfully');
  });
});