import json
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import numpy as np
from .validators import DataQualityChecker, ValidationSeverity

class DataQualityMonitor:
    """Monitor data quality metrics over time and detect degradation"""
    
    def __init__(self, db_path: str = None, alert_thresholds: Dict[str, float] = None):
        self.db_path = db_path or "data_quality_monitor.db"
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
    
    def _default_thresholds(self) -> Dict[str, float]:
        """Default alert thresholds for data quality metrics"""
        return {
            'error_rate': 0.05,  # 5% error rate
            'warning_rate': 0.15,  # 15% warning rate
            'failed_files_rate': 0.10,  # 10% failed files
            'nan_ratio_threshold': 0.01,  # 1% NaN values
            'artifact_ratio_threshold': 0.20,  # 20% artifact epochs
            'variance_threshold': 0.15,  # 15% zero variance epochs
        }
    
    def _init_database(self):
        """Initialize SQLite database for storing quality metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS quality_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    run_timestamp TEXT NOT NULL,
                    total_files INTEGER,
                    passed_files INTEGER,
                    failed_files INTEGER,
                    warning_count INTEGER,
                    error_count INTEGER,
                    metrics_json TEXT,
                    alerts_json TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS file_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    file_path TEXT NOT NULL,
                    validation_status TEXT,
                    nan_ratio REAL,
                    artifact_ratio REAL,
                    variance_ratio REAL,
                    max_amplitude REAL,
                    dc_offset REAL,
                    FOREIGN KEY (run_id) REFERENCES quality_runs (id)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_dataset_timestamp 
                ON quality_runs (dataset_name, run_timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_run_file 
                ON file_metrics (run_id, file_path)
            ''')
    
    def run_quality_check(self, dataset, dataset_name: str) -> Dict[str, Any]:
        """Run quality check and store results"""
        quality_checker = DataQualityChecker()
        quality_report = quality_checker.validate_dataset(dataset)
        
        # Calculate detailed metrics
        detailed_metrics = self._extract_detailed_metrics(quality_report)
        
        # Check for alerts
        alerts = self._check_alerts(detailed_metrics, quality_report)
        
        # Store results in database
        run_id = self._store_quality_run(dataset_name, quality_report, detailed_metrics, alerts)
        self._store_file_metrics(run_id, quality_report)
        
        # Log alerts
        if alerts:
            self._log_alerts(dataset_name, alerts)
        
        return {
            'run_id': run_id,
            'dataset_name': dataset_name,
            'quality_report': quality_report,
            'detailed_metrics': detailed_metrics,
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_detailed_metrics(self, quality_report: Dict[str, Any]) -> Dict[str, float]:
        """Extract detailed metrics from quality report"""
        metrics = {
            'total_files': quality_report['total_files'],
            'passed_files': quality_report['passed_files'],
            'failed_files': quality_report['failed_files'],
            'error_rate': quality_report['errors'] / max(quality_report['total_files'], 1),
            'warning_rate': quality_report['warnings'] / max(quality_report['total_files'], 1),
            'pass_rate': quality_report['passed_files'] / max(quality_report['total_files'], 1),
        }
        
        # Aggregate file-level metrics
        nan_ratios = []
        artifact_ratios = []
        variance_ratios = []
        max_amplitudes = []
        dc_offsets = []
        
        for file_result in quality_report['validation_results']:
            for result in file_result['results']:
                if result['metadata']:
                    if result['test_name'] == 'nan_check' and 'nan_ratio' in result['metadata']:
                        nan_ratios.append(result['metadata']['nan_ratio'])
                    elif result['test_name'] == 'artifact_detection' and 'artifact_ratio' in result['metadata']:
                        artifact_ratios.append(result['metadata']['artifact_ratio'])
                    elif result['test_name'] == 'signal_variance' and 'zero_var_ratio' in result['metadata']:
                        variance_ratios.append(result['metadata']['zero_var_ratio'])
                    elif result['test_name'] == 'amplitude_range' and 'max_amplitude' in result['metadata']:
                        max_amplitudes.append(result['metadata']['max_amplitude'])
                    elif result['test_name'] == 'dc_offset' and 'dc_offset' in result['metadata']:
                        dc_offsets.append(abs(result['metadata']['dc_offset']))
        
        # Calculate aggregate statistics
        if nan_ratios:
            metrics['avg_nan_ratio'] = np.mean(nan_ratios)
            metrics['max_nan_ratio'] = np.max(nan_ratios)
        
        if artifact_ratios:
            metrics['avg_artifact_ratio'] = np.mean(artifact_ratios)
            metrics['max_artifact_ratio'] = np.max(artifact_ratios)
        
        if variance_ratios:
            metrics['avg_variance_ratio'] = np.mean(variance_ratios)
            metrics['max_variance_ratio'] = np.max(variance_ratios)
        
        if max_amplitudes:
            metrics['avg_max_amplitude'] = np.mean(max_amplitudes)
            metrics['max_amplitude'] = np.max(max_amplitudes)
        
        if dc_offsets:
            metrics['avg_dc_offset'] = np.mean(dc_offsets)
            metrics['max_dc_offset'] = np.max(dc_offsets)
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, float], quality_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for quality degradation alerts"""
        alerts = []
        
        # Check error rate
        if metrics.get('error_rate', 0) > self.alert_thresholds['error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'message': f"Error rate {metrics['error_rate']:.3f} exceeds threshold {self.alert_thresholds['error_rate']}",
                'value': metrics['error_rate'],
                'threshold': self.alert_thresholds['error_rate']
            })
        
        # Check warning rate
        if metrics.get('warning_rate', 0) > self.alert_thresholds['warning_rate']:
            alerts.append({
                'type': 'high_warning_rate',
                'severity': 'warning',
                'message': f"Warning rate {metrics['warning_rate']:.3f} exceeds threshold {self.alert_thresholds['warning_rate']}",
                'value': metrics['warning_rate'],
                'threshold': self.alert_thresholds['warning_rate']
            })
        
        # Check NaN ratio
        if metrics.get('max_nan_ratio', 0) > self.alert_thresholds['nan_ratio_threshold']:
            alerts.append({
                'type': 'high_nan_ratio',
                'severity': 'critical',
                'message': f"Maximum NaN ratio {metrics['max_nan_ratio']:.3f} exceeds threshold {self.alert_thresholds['nan_ratio_threshold']}",
                'value': metrics['max_nan_ratio'],
                'threshold': self.alert_thresholds['nan_ratio_threshold']
            })
        
        # Check artifact ratio
        if metrics.get('max_artifact_ratio', 0) > self.alert_thresholds['artifact_ratio_threshold']:
            alerts.append({
                'type': 'high_artifact_ratio',
                'severity': 'warning',
                'message': f"Maximum artifact ratio {metrics['max_artifact_ratio']:.3f} exceeds threshold {self.alert_thresholds['artifact_ratio_threshold']}",
                'value': metrics['max_artifact_ratio'],
                'threshold': self.alert_thresholds['artifact_ratio_threshold']
            })
        
        return alerts
    
    def _store_quality_run(self, dataset_name: str, quality_report: Dict[str, Any], 
                          metrics: Dict[str, float], alerts: List[Dict[str, Any]]) -> int:
        """Store quality run in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO quality_runs 
                (dataset_name, run_timestamp, total_files, passed_files, failed_files, 
                 warning_count, error_count, metrics_json, alerts_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                dataset_name,
                datetime.now().isoformat(),
                quality_report['total_files'],
                quality_report['passed_files'],
                quality_report['failed_files'],
                quality_report['warnings'],
                quality_report['errors'],
                json.dumps(metrics),
                json.dumps(alerts)
            ))
            return cursor.lastrowid
    
    def _store_file_metrics(self, run_id: int, quality_report: Dict[str, Any]):
        """Store individual file metrics"""
        with sqlite3.connect(self.db_path) as conn:
            for file_result in quality_report['validation_results']:
                # Extract metrics for this file
                file_metrics = {
                    'nan_ratio': None,
                    'artifact_ratio': None,
                    'variance_ratio': None,
                    'max_amplitude': None,
                    'dc_offset': None
                }
                
                for result in file_result['results']:
                    if result['metadata']:
                        if result['test_name'] == 'nan_check' and 'nan_ratio' in result['metadata']:
                            file_metrics['nan_ratio'] = result['metadata']['nan_ratio']
                        elif result['test_name'] == 'artifact_detection' and 'artifact_ratio' in result['metadata']:
                            file_metrics['artifact_ratio'] = result['metadata']['artifact_ratio']
                        elif result['test_name'] == 'signal_variance' and 'zero_var_ratio' in result['metadata']:
                            file_metrics['variance_ratio'] = result['metadata']['zero_var_ratio']
                        elif result['test_name'] == 'amplitude_range' and 'max_amplitude' in result['metadata']:
                            file_metrics['max_amplitude'] = result['metadata']['max_amplitude']
                        elif result['test_name'] == 'dc_offset' and 'dc_offset' in result['metadata']:
                            file_metrics['dc_offset'] = result['metadata']['dc_offset']
                
                conn.execute('''
                    INSERT INTO file_metrics 
                    (run_id, file_path, validation_status, nan_ratio, artifact_ratio, 
                     variance_ratio, max_amplitude, dc_offset)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id,
                    file_result['file_path'],
                    file_result['status'],
                    file_metrics['nan_ratio'],
                    file_metrics['artifact_ratio'],
                    file_metrics['variance_ratio'],
                    file_metrics['max_amplitude'],
                    file_metrics['dc_offset']
                ))
    
    def _log_alerts(self, dataset_name: str, alerts: List[Dict[str, Any]]):
        """Log alerts"""
        for alert in alerts:
            severity_emoji = "🚨" if alert['severity'] == 'critical' else "⚠️"
            self.logger.warning(f"{severity_emoji} {dataset_name}: {alert['message']}")
    
    def get_quality_trends(self, dataset_name: str, days: int = 30) -> Dict[str, Any]:
        """Get quality trends for the last N days"""
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT run_timestamp, total_files, passed_files, failed_files, 
                       warning_count, error_count, metrics_json
                FROM quality_runs 
                WHERE dataset_name = ? AND run_timestamp >= ?
                ORDER BY run_timestamp
            ''', (dataset_name, since_date))
            
            runs = cursor.fetchall()
        
        if not runs:
            return {'message': f'No quality data found for {dataset_name} in last {days} days'}
        
        # Process trends
        timestamps = []
        error_rates = []
        pass_rates = []
        
        for run in runs:
            timestamps.append(run[0])
            total_files = run[1]
            passed_files = run[2]
            metrics = json.loads(run[6]) if run[6] else {}
            
            error_rates.append(metrics.get('error_rate', 0))
            pass_rates.append(passed_files / max(total_files, 1))
        
        return {
            'dataset_name': dataset_name,
            'period_days': days,
            'total_runs': len(runs),
            'timestamps': timestamps,
            'trends': {
                'error_rate': {
                    'values': error_rates,
                    'current': error_rates[-1] if error_rates else 0,
                    'average': np.mean(error_rates) if error_rates else 0,
                    'trend': 'improving' if len(error_rates) > 1 and error_rates[-1] < error_rates[0] else 'declining'
                },
                'pass_rate': {
                    'values': pass_rates,
                    'current': pass_rates[-1] if pass_rates else 0,
                    'average': np.mean(pass_rates) if pass_rates else 0,
                    'trend': 'improving' if len(pass_rates) > 1 and pass_rates[-1] > pass_rates[0] else 'declining'
                }
            }
        }
    
    def get_recent_alerts(self, dataset_name: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        since_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        query = '''
            SELECT dataset_name, run_timestamp, alerts_json
            FROM quality_runs 
            WHERE run_timestamp >= ? AND alerts_json != '[]' AND alerts_json IS NOT NULL
        '''
        params = [since_time]
        
        if dataset_name:
            query += ' AND dataset_name = ?'
            params.append(dataset_name)
        
        query += ' ORDER BY run_timestamp DESC'
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            runs = cursor.fetchall()
        
        recent_alerts = []
        for run in runs:
            alerts = json.loads(run[2]) if run[2] else []
            for alert in alerts:
                alert['dataset_name'] = run[0]
                alert['timestamp'] = run[1]
                recent_alerts.append(alert)
        
        return recent_alerts