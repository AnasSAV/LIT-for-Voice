import sqlite3
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiosqlite

logger = logging.getLogger(__name__)

class PredictionDatabase:
    """
    SQLite database service for persisting prediction results across sessions
    """
    
    def __init__(self, db_path: str = "predictions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
    async def initialize(self):
        """Initialize the database with required tables"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    model TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    metadata TEXT,
                    predictions_data TEXT NOT NULL,
                    file_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS prediction_files (
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
                )
            """)
            
            # Create indexes for better performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_cache_key ON predictions (cache_key)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_model_dataset ON predictions (model, dataset)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_prediction_id ON prediction_files (prediction_id)")
            
            await db.commit()
            logger.info("Database initialized successfully")
    
    def _generate_cache_key(self, model: str, dataset: str, **kwargs) -> str:
        """Generate a unique cache key for model-dataset combination"""
        key_data = {
            "model": model,
            "dataset": dataset,
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def save_predictions(self, model: str, dataset: str, data: Dict, **kwargs) -> int:
        """
        Save prediction results to database
        Returns the prediction ID
        """
        cache_key = self._generate_cache_key(model, dataset, **kwargs)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Insert or update main prediction record
            await db.execute("""
                INSERT OR REPLACE INTO predictions 
                (cache_key, model, dataset, metadata, predictions_data, file_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                model,
                dataset,
                json.dumps(kwargs),
                json.dumps(data),
                len(data.get('predictions', [])),
                datetime.now().isoformat()
            ))
            
            # Get the prediction ID
            cursor = await db.execute("SELECT id FROM predictions WHERE cache_key = ?", (cache_key,))
            row = await cursor.fetchone()
            prediction_id = row[0] if row else None
            
            if prediction_id and 'predictions' in data:
                # Delete existing prediction files for this prediction
                await db.execute("DELETE FROM prediction_files WHERE prediction_id = ?", (prediction_id,))
                
                # Insert individual prediction files
                for pred in data['predictions']:
                    await db.execute("""
                        INSERT INTO prediction_files 
                        (prediction_id, file_path, file_name, emotion_prediction, emotion_confidence, 
                         transcription, ground_truth_emotion, ground_truth_transcript)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        prediction_id,
                        pred.get('file_path', ''),
                        pred.get('file_name', ''),
                        pred.get('emotion_prediction', ''),
                        pred.get('emotion_confidence', 0.0),
                        pred.get('transcription', ''),
                        pred.get('ground_truth_emotion', ''),
                        pred.get('ground_truth_transcript', '')
                    ))
            
            await db.commit()
            logger.info(f"Saved predictions to database: {model}-{dataset} ({len(data.get('predictions', []))} files)")
            return prediction_id
    
    async def get_predictions(self, model: str, dataset: str, **kwargs) -> Optional[Dict]:
        """
        Retrieve prediction results from database
        """
        cache_key = self._generate_cache_key(model, dataset, **kwargs)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Get main prediction record
            cursor = await db.execute("""
                SELECT id, predictions_data, created_at, updated_at 
                FROM predictions 
                WHERE cache_key = ?
            """, (cache_key,))
            
            row = await cursor.fetchone()
            if not row:
                return None
            
            prediction_id, predictions_data, created_at, updated_at = row
            
            try:
                data = json.loads(predictions_data)
                data['cache_info'] = {
                    'created_at': created_at,
                    'updated_at': updated_at,
                    'source': 'database'
                }
                logger.info(f"Retrieved predictions from database: {model}-{dataset}")
                return data
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing prediction data from database: {e}")
                return None
    
    async def list_predictions(self, model: str = None, dataset: str = None) -> List[Dict]:
        """
        List all predictions, optionally filtered by model and/or dataset
        """
        query = "SELECT model, dataset, file_count, created_at, updated_at FROM predictions"
        params = []
        
        conditions = []
        if model:
            conditions.append("model = ?")
            params.append(model)
        if dataset:
            conditions.append("dataset = ?")
            params.append(dataset)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY updated_at DESC"
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            
            return [
                {
                    'model': row[0],
                    'dataset': row[1],
                    'file_count': row[2],
                    'created_at': row[3],
                    'updated_at': row[4]
                }
                for row in rows
            ]
    
    async def delete_predictions(self, model: str = None, dataset: str = None) -> int:
        """
        Delete predictions, optionally filtered by model and/or dataset
        Returns number of deleted records
        """
        conditions = []
        params = []
        
        if model:
            conditions.append("model = ?")
            params.append(model)
        if dataset:
            conditions.append("dataset = ?")
            params.append(dataset)
        
        if not conditions:
            # Delete all predictions
            query = "DELETE FROM predictions"
        else:
            query = f"DELETE FROM predictions WHERE {' AND '.join(conditions)}"
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            deleted_count = cursor.rowcount
            await db.commit()
            
            logger.info(f"Deleted {deleted_count} prediction records from database")
            return deleted_count
    
    async def get_database_stats(self) -> Dict:
        """Get database statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            # Count predictions
            cursor = await db.execute("SELECT COUNT(*) FROM predictions")
            prediction_count = (await cursor.fetchone())[0]
            
            # Count prediction files
            cursor = await db.execute("SELECT COUNT(*) FROM prediction_files")
            file_count = (await cursor.fetchone())[0]
            
            # Get database file size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            # Get models and datasets
            cursor = await db.execute("SELECT DISTINCT model FROM predictions")
            models = [row[0] for row in await cursor.fetchall()]
            
            cursor = await db.execute("SELECT DISTINCT dataset FROM predictions")
            datasets = [row[0] for row in await cursor.fetchall()]
            
            return {
                'prediction_records': prediction_count,
                'total_files': file_count,
                'database_size_bytes': db_size,
                'database_size_mb': round(db_size / 1024 / 1024, 2),
                'database_path': str(self.db_path.absolute()),
                'models': models,
                'datasets': datasets
            }
    
    async def export_to_sql(self, output_path: str = None) -> str:
        """
        Export database to SQL file
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"predictions_export_{timestamp}.sql"
        
        output_file = Path(output_path)
        
        async with aiosqlite.connect(self.db_path) as db:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write schema
                f.write("-- Predictions Database Export\n")
                f.write(f"-- Generated: {datetime.now().isoformat()}\n\n")
                
                # Export schema
                cursor = await db.execute("SELECT sql FROM sqlite_master WHERE type='table'")
                for row in await cursor.fetchall():
                    if row[0]:
                        f.write(f"{row[0]};\n\n")
                
                # Export data
                f.write("-- Data Export\n\n")
                
                # Export predictions table
                cursor = await db.execute("SELECT * FROM predictions")
                columns = [description[0] for description in cursor.description]
                
                for row in await cursor.fetchall():
                    values = []
                    for value in row:
                        if value is None:
                            values.append("NULL")
                        elif isinstance(value, str):
                            # Escape single quotes
                            escaped_value = value.replace("'", "''")
                            values.append(f"'{escaped_value}'")
                        else:
                            values.append(str(value))
                    
                    f.write(f"INSERT INTO predictions ({', '.join(columns)}) VALUES ({', '.join(values)});\n")
                
                f.write("\n")
                
                # Export prediction_files table
                cursor = await db.execute("SELECT * FROM prediction_files")
                columns = [description[0] for description in cursor.description]
                
                for row in await cursor.fetchall():
                    values = []
                    for value in row:
                        if value is None:
                            values.append("NULL")
                        elif isinstance(value, str):
                            escaped_value = value.replace("'", "''")
                            values.append(f"'{escaped_value}'")
                        else:
                            values.append(str(value))
                    
                    f.write(f"INSERT INTO prediction_files ({', '.join(columns)}) VALUES ({', '.join(values)});\n")
        
        logger.info(f"Database exported to {output_file}")
        return str(output_file)

# Global database instance
prediction_db = PredictionDatabase()
