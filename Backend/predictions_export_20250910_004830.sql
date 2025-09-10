-- Predictions Database Export
-- Generated: 2025-09-10T00:48:30.881981

CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    model TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    metadata TEXT,
                    predictions_data TEXT NOT NULL,
                    file_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

CREATE TABLE sqlite_sequence(name,seq);

CREATE TABLE prediction_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    emotion_prediction TEXT,
                    emotion_confidence REAL,
                    transcription TEXT,
                    ground_truth_emotion TEXT,
                    ground_truth_transcript TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prediction_id) REFERENCES predictions (id) ON DELETE CASCADE
                );

-- Data Export


