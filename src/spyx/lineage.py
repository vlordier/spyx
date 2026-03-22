"""
Experiment Lineage Analytics for Autonomous SNN Research

Provides deep experiment tracking with Git-aware lineage, parameter ablations,
and trend analysis. Combines JSONL logs + optional SQLite backend for structured queries.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


@dataclass
class ExperimentMetadata:
    """Experiment metadata with Git and reproducibility info."""
    experiment_id: str
    timestamp: str
    git_commit: str = ""
    git_branch: str = "main"
    config_hash: str = ""
    parent_experiment_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class ExperimentLineageDB:
    """
    SQLite database for experiment tracking and advanced querying.
    Maintains lineage relationships between experiments (e.g., ablation studies).
    """

    def __init__(self, db_path: str = "experiments.db"):
        """Initialize or open SQLite database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                git_commit TEXT,
                git_branch TEXT,
                config_hash TEXT,
                parent_experiment_id TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Metrics table (normalized for querying)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                step INTEGER DEFAULT 0,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
                UNIQUE(experiment_id, metric_name, step)
            )
        """)
        
        # Config parameters table (normalized for ablation analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config_params (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                param_name TEXT NOT NULL,
                param_value TEXT NOT NULL,
                param_type TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
                UNIQUE(experiment_id, param_name)
            )
        """)
        
        # Ablation studies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ablations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                base_experiment_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                parameter_name TEXT NOT NULL,
                base_value TEXT,
                variant_value TEXT,
                base_metric REAL,
                variant_metric REAL,
                delta REAL,
                FOREIGN KEY (base_experiment_id) REFERENCES experiments(experiment_id),
                FOREIGN KEY (variant_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        self.conn.commit()

    def insert_experiment(self, metadata: ExperimentMetadata, config: Dict[str, Any], 
                         metrics: Dict[str, Any]) -> None:
        """Insert experiment record."""
        cursor = self.conn.cursor()
        
        # Insert experiment metadata
        cursor.execute("""
            INSERT OR REPLACE INTO experiments 
            (experiment_id, timestamp, git_commit, git_branch, config_hash, 
             parent_experiment_id, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.experiment_id,
            metadata.timestamp,
            metadata.git_commit,
            metadata.git_branch,
            metadata.config_hash,
            metadata.parent_experiment_id,
            json.dumps(metadata.tags),
        ))
        
        # Insert config parameters
        for param_name, param_value in config.items():
            param_type = type(param_value).__name__
            cursor.execute("""
                INSERT OR REPLACE INTO config_params 
                (experiment_id, param_name, param_value, param_type)
                VALUES (?, ?, ?, ?)
            """, (
                metadata.experiment_id,
                param_name,
                str(param_value),
                param_type,
            ))
        
        # Insert metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                cursor.execute("""
                    INSERT OR REPLACE INTO metrics
                    (experiment_id, metric_name, metric_value)
                    VALUES (?, ?, ?)
                """, (
                    metadata.experiment_id,
                    metric_name,
                    float(metric_value),
                ))
        
        self.conn.commit()

    def query_experiments_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Query experiments by tag."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT experiment_id, timestamp, git_commit, config_hash
            FROM experiments
            WHERE tags LIKE ?
            ORDER BY timestamp DESC
        """, (f'%"{tag}"%',))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "experiment_id": row[0],
                "timestamp": row[1],
                "git_commit": row[2],
                "config_hash": row[3],
            })
        return results

    def query_metric_trend(self, metric_name: str, param_name: str, 
                          tag: Optional[str] = None) -> List[Tuple]:
        """Query how a metric varies with a config parameter."""
        cursor = self.conn.cursor()
        
        query = """
            SELECT cp.param_value, m.metric_value, e.timestamp
            FROM config_params cp
            JOIN metrics m ON cp.experiment_id = m.experiment_id
            JOIN experiments e ON cp.experiment_id = e.experiment_id
            WHERE cp.param_name = ? AND m.metric_name = ?
        """
        params = [param_name, metric_name]
        
        if tag:
            query += " AND e.tags LIKE ?"
            params.append(f'%"{tag}"%')
        
        query += " ORDER BY cp.param_value, e.timestamp"
        
        cursor.execute(query, params)
        return cursor.fetchall()

    def compute_ablation(self, base_exp_id: str, variants: List[Tuple[str, Dict]],
                        metric_name: str = "accuracy") -> List[Dict]:
        """
        Analyze ablation study: compute metric delta for parameter variations.
        
        Args:
            base_exp_id: Base experiment ID
            variants: List of (variant_exp_id, config_diff) tuples
            metric_name: Metric to compare (e.g., "accuracy")
        
        Returns:
            List of ablation records with delta calculations
        """
        cursor = self.conn.cursor()
        
        # Get base metric
        cursor.execute("""
            SELECT metric_value FROM metrics
            WHERE experiment_id = ? AND metric_name = ?
        """, (base_exp_id, metric_name))
        
        base_result = cursor.fetchone()
        if not base_result:
            return []
        
        base_metric = base_result[0]
        ablations = []
        
        for variant_id, config_diff in variants:
            cursor.execute("""
                SELECT metric_value FROM metrics
                WHERE experiment_id = ? AND metric_name = ?
            """, (variant_id, metric_name))
            
            variant_result = cursor.fetchone()
            if not variant_result:
                continue
            
            variant_metric = variant_result[0]
            delta = variant_metric - base_metric
            
            # Get parameter that changed
            for param_name, variant_value in config_diff.items():
                cursor.execute("""
                    SELECT param_value FROM config_params
                    WHERE experiment_id = ? AND param_name = ?
                """, (base_exp_id, param_name))
                
                base_result = cursor.fetchone()
                base_value = base_result[0] if base_result else None
                
                ablations.append({
                    "parameter_name": param_name,
                    "base_value": base_value,
                    "variant_value": str(variant_value),
                    "base_metric": base_metric,
                    "variant_metric": variant_metric,
                    "delta": delta,
                    "relative_delta": (delta / abs(base_metric)) * 100 if base_metric != 0 else 0,
                })
        
        return ablations

    def get_experiment_lineage(self, experiment_id: str, max_depth: int = 5) -> Dict:
        """Get full experiment lineage (parents and children)."""
        cursor = self.conn.cursor()
        
        lineage = {"experiment_id": experiment_id, "parents": [], "children": []}
        
        # Get parents
        current_id = experiment_id
        for _ in range(max_depth):
            cursor.execute("""
                SELECT parent_experiment_id FROM experiments
                WHERE experiment_id = ?
            """, (current_id,))
            
            result = cursor.fetchone()
            if result and result[0]:
                lineage["parents"].append(result[0])
                current_id = result[0]
            else:
                break
        
        # Get children
        cursor.execute("""
            SELECT experiment_id FROM experiments
            WHERE parent_experiment_id = ?
        """, (experiment_id,))
        
        lineage["children"] = [row[0] for row in cursor.fetchall()]
        
        return lineage

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of all experiments."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM experiments")
        total_experiments = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT git_commit) FROM experiments WHERE git_commit != ''")
        unique_commits = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT git_branch) FROM experiments WHERE git_branch != ''")
        unique_branches = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(metric_value) FROM metrics WHERE metric_name = 'accuracy'")
        avg_accuracy = cursor.fetchone()[0] or 0.0
        
        return {
            "total_experiments": total_experiments,
            "unique_git_commits": unique_commits,
            "unique_branches": unique_branches,
            "average_accuracy": avg_accuracy,
        }

    def close(self):
        """Close database connection."""
        self.conn.close()


class ExperimentAnalyzer:
    """Analyze experiments from JSONL logs."""

    def __init__(self, jsonl_path: str = "experiments.jsonl"):
        """Initialize with JSONL log file path."""
        self.jsonl_path = Path(jsonl_path)
        self.experiments = []
        self._load_experiments()

    def _load_experiments(self):
        """Load all experiments from JSONL file."""
        if not self.jsonl_path.exists():
            return
        
        with open(self.jsonl_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    self.experiments.append(entry)
                except json.JSONDecodeError:
                    continue

    def compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash of configuration for tracking changes."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def find_best_experiment(self, metric_name: str = "accuracy") -> Optional[Dict]:
        """Find experiment with best metric value."""
        if not self.experiments:
            return None
        
        best = None
        best_value = float("-inf")
        
        for exp in self.experiments:
            metrics = exp.get("metrics", {})
            if metric_name in metrics:
                value = metrics[metric_name]
                if value > best_value:
                    best_value = value
                    best = exp
        
        return best

    def analyze_learning_curves(self) -> Dict[str, List[float]]:
        """Extract learning curves (if available in metrics)."""
        curves = {}
        
        for exp in self.experiments:
            exp_id = exp.get("experiment_id", "unknown")
            metrics = exp.get("metrics", {})
            
            # Look for epoch-wise metrics
            for key, value in metrics.items():
                if "loss" in key.lower() or "accuracy" in key.lower():
                    if key not in curves:
                        curves[key] = []
                    curves[key].append(value)
        
        return curves

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate markdown report of experiments."""
        report = []
        report.append("# Experiment Lineage Report\n")
        report.append(f"Generated: {datetime.now().isoformat()}\n")
        report.append(f"Total experiments: {len(self.experiments)}\n\n")
        
        # Best experiment
        best = self.find_best_experiment("accuracy")
        if best:
            report.append("## Best Experiment\n")
            report.append(f"ID: `{best.get('experiment_id', 'N/A')}`\n")
            report.append(f"Accuracy: {best.get('metrics', {}).get('accuracy', 'N/A')}\n")
            report.append(f"Config: {json.dumps(best.get('config', {}), indent=2)}\n\n")
        
        # Summary
        report.append("## Experiment Summary\n")
        report.append("| Experiment ID | Accuracy | Training Time |\n")
        report.append("|---|---|---|\n")
        for exp in sorted(self.experiments, key=lambda x: x.get("metrics", {}).get("accuracy", 0), reverse=True)[:10]:
            exp_id = exp.get("experiment_id", "N/A")
            acc = exp.get("metrics", {}).get("accuracy", "N/A")
            duration = exp.get("metrics", {}).get("duration", "N/A")
            report.append(f"| `{exp_id}` | {acc:.4f} | {duration:.2f}s |\n")
        
        report_text = "".join(report)
        
        if output_file:
            Path(output_file).write_text(report_text)
        
        return report_text
