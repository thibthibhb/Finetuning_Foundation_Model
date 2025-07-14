import json
import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import inspect

class DataLineageTracker:
    """Track data transformations and provenance for EEG datasets"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or "data_lineage.db"
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
        # Track current transformation context
        self.current_transformation = None
    
    def _init_database(self):
        """Initialize SQLite database for lineage tracking"""
        with sqlite3.connect(self.db_path) as conn:
            # Dataset versions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    path TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT
                )
            ''')
            
            # Transformations table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS transformations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    function_name TEXT,
                    parameters_json TEXT,
                    created_at TEXT NOT NULL,
                    execution_time_ms REAL,
                    status TEXT
                )
            ''')
            
            # Lineage edges table (what produced what)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS lineage_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transformation_id TEXT NOT NULL,
                    input_dataset_id TEXT NOT NULL,
                    output_dataset_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (transformation_id) REFERENCES transformations (id),
                    FOREIGN KEY (input_dataset_id) REFERENCES dataset_versions (id),
                    FOREIGN KEY (output_dataset_id) REFERENCES dataset_versions (id)
                )
            ''')
            
            # Quality metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS dataset_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    recorded_at TEXT NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES dataset_versions (id)
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_lineage_transformation ON lineage_edges (transformation_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_lineage_input ON lineage_edges (input_dataset_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_lineage_output ON lineage_edges (output_dataset_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_quality_dataset ON dataset_quality (dataset_id)')
    
    def register_dataset_version(self, name: str, version: str, path: str, 
                                metadata: Dict[str, Any] = None) -> str:
        """Register a new dataset version"""
        # Calculate dataset hash
        dataset_hash = self._calculate_path_hash(path)
        
        # Create unique ID
        dataset_id = f"{name}_{version}_{dataset_hash[:8]}"
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO dataset_versions 
                (id, name, version, path, hash, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                dataset_id,
                name,
                version,
                path,
                dataset_hash,
                datetime.now().isoformat(),
                json.dumps(metadata or {})
            ))
        
        self.logger.info(f"Registered dataset version: {dataset_id}")
        return dataset_id
    
    def start_transformation(self, name: str, description: str = "", 
                           function_name: str = "", parameters: Dict[str, Any] = None) -> str:
        """Start tracking a data transformation"""
        transformation_id = hashlib.md5(
            f"{name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Get caller function name if not provided
        if not function_name:
            frame = inspect.currentframe().f_back
            function_name = frame.f_code.co_name
        
        transformation_data = {
            'id': transformation_id,
            'name': name,
            'description': description,
            'function_name': function_name,
            'parameters': parameters or {},
            'started_at': datetime.now().isoformat(),
            'status': 'running'
        }
        
        self.current_transformation = transformation_data
        self.logger.info(f"Started transformation: {name} ({transformation_id})")
        return transformation_id
    
    def end_transformation(self, transformation_id: str, status: str = "completed", 
                          execution_time_ms: float = None):
        """End tracking a data transformation"""
        if self.current_transformation and self.current_transformation['id'] == transformation_id:
            # Calculate execution time if not provided
            if execution_time_ms is None:
                start_time = datetime.fromisoformat(self.current_transformation['started_at'])
                execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Store transformation in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO transformations 
                    (id, name, description, function_name, parameters_json, 
                     created_at, execution_time_ms, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transformation_id,
                    self.current_transformation['name'],
                    self.current_transformation['description'],
                    self.current_transformation['function_name'],
                    json.dumps(self.current_transformation['parameters']),
                    self.current_transformation['started_at'],
                    execution_time_ms,
                    status
                ))
            
            self.logger.info(f"Completed transformation: {transformation_id} ({status})")
            self.current_transformation = None
    
    def record_lineage(self, transformation_id: str, input_dataset_id: str, 
                      output_dataset_id: str):
        """Record lineage relationship between datasets"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO lineage_edges 
                (transformation_id, input_dataset_id, output_dataset_id, created_at)
                VALUES (?, ?, ?, ?)
            ''', (
                transformation_id,
                input_dataset_id,
                output_dataset_id,
                datetime.now().isoformat()
            ))
        
        self.logger.debug(f"Recorded lineage: {input_dataset_id} -> {output_dataset_id} via {transformation_id}")
    
    def record_quality_metric(self, dataset_id: str, metric_name: str, metric_value: float):
        """Record a quality metric for a dataset"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO dataset_quality 
                (dataset_id, metric_name, metric_value, recorded_at)
                VALUES (?, ?, ?, ?)
            ''', (
                dataset_id,
                metric_name,
                metric_value,
                datetime.now().isoformat()
            ))
    
    def get_dataset_lineage(self, dataset_id: str, depth: int = 3) -> Dict[str, Any]:
        """Get the lineage graph for a dataset"""
        lineage = {
            'dataset_id': dataset_id,
            'upstream': [],
            'downstream': [],
            'transformations': {}
        }
        
        # Get upstream lineage
        upstream = self._get_upstream_lineage(dataset_id, depth)
        lineage['upstream'] = upstream
        
        # Get downstream lineage  
        downstream = self._get_downstream_lineage(dataset_id, depth)
        lineage['downstream'] = downstream
        
        # Get transformation details
        all_transformation_ids = set()
        for item in upstream + downstream:
            if 'transformation_id' in item:
                all_transformation_ids.add(item['transformation_id'])
        
        lineage['transformations'] = self._get_transformation_details(list(all_transformation_ids))
        
        return lineage
    
    def _get_upstream_lineage(self, dataset_id: str, depth: int) -> List[Dict[str, Any]]:
        """Get datasets that contributed to this dataset"""
        if depth <= 0:
            return []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT le.transformation_id, le.input_dataset_id, dv.name, dv.version, dv.created_at
                FROM lineage_edges le
                JOIN dataset_versions dv ON le.input_dataset_id = dv.id
                WHERE le.output_dataset_id = ?
                ORDER BY le.created_at
            ''', (dataset_id,))
            
            upstream = []
            for row in cursor.fetchall():
                transformation_id, input_id, name, version, created_at = row
                
                upstream_item = {
                    'dataset_id': input_id,
                    'name': name,
                    'version': version,
                    'created_at': created_at,
                    'transformation_id': transformation_id,
                    'upstream': self._get_upstream_lineage(input_id, depth - 1)
                }
                upstream.append(upstream_item)
            
            return upstream
    
    def _get_downstream_lineage(self, dataset_id: str, depth: int) -> List[Dict[str, Any]]:
        """Get datasets that were derived from this dataset"""
        if depth <= 0:
            return []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT le.transformation_id, le.output_dataset_id, dv.name, dv.version, dv.created_at
                FROM lineage_edges le
                JOIN dataset_versions dv ON le.output_dataset_id = dv.id
                WHERE le.input_dataset_id = ?
                ORDER BY le.created_at
            ''', (dataset_id,))
            
            downstream = []
            for row in cursor.fetchall():
                transformation_id, output_id, name, version, created_at = row
                
                downstream_item = {
                    'dataset_id': output_id,
                    'name': name,
                    'version': version,
                    'created_at': created_at,
                    'transformation_id': transformation_id,
                    'downstream': self._get_downstream_lineage(output_id, depth - 1)
                }
                downstream.append(downstream_item)
            
            return downstream
    
    def _get_transformation_details(self, transformation_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get details for transformations"""
        if not transformation_ids:
            return {}
        
        placeholders = ','.join(['?' for _ in transformation_ids])
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f'''
                SELECT id, name, description, function_name, parameters_json, 
                       created_at, execution_time_ms, status
                FROM transformations
                WHERE id IN ({placeholders})
            ''', transformation_ids)
            
            transformations = {}
            for row in cursor.fetchall():
                t_id, name, description, function_name, params_json, created_at, exec_time, status = row
                
                transformations[t_id] = {
                    'name': name,
                    'description': description,
                    'function_name': function_name,
                    'parameters': json.loads(params_json) if params_json else {},
                    'created_at': created_at,
                    'execution_time_ms': exec_time,
                    'status': status
                }
            
            return transformations
    
    def get_dataset_quality_history(self, dataset_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get quality metric history for a dataset"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT metric_name, metric_value, recorded_at
                FROM dataset_quality
                WHERE dataset_id = ?
                ORDER BY metric_name, recorded_at
            ''', (dataset_id,))
            
            quality_history = {}
            for row in cursor.fetchall():
                metric_name, metric_value, recorded_at = row
                
                if metric_name not in quality_history:
                    quality_history[metric_name] = []
                
                quality_history[metric_name].append({
                    'value': metric_value,
                    'recorded_at': recorded_at
                })
            
            return quality_history
    
    def _calculate_path_hash(self, path: str) -> str:
        """Calculate hash for a path (simplified)"""
        path_obj = Path(path)
        
        if path_obj.is_file():
            # Hash file content
            with open(path_obj, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        elif path_obj.is_dir():
            # Hash directory structure and file stats
            hasher = hashlib.md5()
            for file_path in sorted(path_obj.rglob('*')):
                if file_path.is_file():
                    stat = file_path.stat()
                    hasher.update(f"{file_path.name}:{stat.st_size}:{stat.st_mtime}".encode())
            return hasher.hexdigest()
        else:
            # Hash the path string itself
            return hashlib.md5(path.encode()).hexdigest()
    
    def export_lineage_graph(self, dataset_id: str, output_path: str):
        """Export lineage graph to JSON for visualization"""
        lineage = self.get_dataset_lineage(dataset_id)
        
        with open(output_path, 'w') as f:
            json.dump(lineage, f, indent=2, default=str)
        
        self.logger.info(f"Exported lineage graph to {output_path}")
    
    def get_transformation_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of transformations in the last N days"""
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT COUNT(*) as total,
                       AVG(execution_time_ms) as avg_time,
                       SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                       SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                FROM transformations
                WHERE created_at >= ?
            ''', (since_date,))
            
            row = cursor.fetchone()
            
            return {
                'period_days': days,
                'total_transformations': row[0] or 0,
                'avg_execution_time_ms': row[1] or 0,
                'completed': row[2] or 0,
                'failed': row[3] or 0,
                'success_rate': (row[2] or 0) / max(row[0] or 1, 1)
            }