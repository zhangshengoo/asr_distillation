"""Fault tolerance and monitoring system"""

import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import psutil
import ray

from prometheus_client import Counter, Histogram, Gauge, start_http_server
from loguru import logger
from src.config.manager import MonitoringConfig


class MetricsCollector:
    """Prometheus metrics collector"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Define metrics
        self.processed_batches = Counter('asr_processed_batches_total', 'Total processed batches')
        self.processed_items = Counter('asr_processed_items_total', 'Total processed items')
        self.error_count = Counter('asr_errors_total', 'Total errors', ['stage', 'error_type'])
        self.processing_time = Histogram('asr_processing_duration_seconds', 'Processing time', ['stage'])
        self.queue_size = Gauge('asr_queue_size', 'Queue size', ['queue_name'])
        self.gpu_utilization = Gauge('asr_gpu_utilization_percent', 'GPU utilization', ['gpu_id'])
        self.gpu_memory_used = Gauge('asr_gpu_memory_used_bytes', 'GPU memory used', ['gpu_id'])
        self.gpu_memory_total = Gauge('asr_gpu_memory_total_bytes', 'GPU memory total', ['gpu_id'])
        self.cpu_utilization = Gauge('asr_cpu_utilization_percent', 'CPU utilization')
        self.memory_usage = Gauge('asr_memory_usage_bytes', 'Memory usage')
        self.object_store_usage = Gauge('asr_object_store_usage_bytes', 'Ray object store usage')
        
        if config.enable_prometheus:
            start_http_server(config.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {config.prometheus_port}")
    
    def record_batch_processed(self, stage: str, processing_time: float, item_count: int) -> None:
        """Record batch processing metrics"""
        self.processed_batches.inc()
        self.processed_items.inc(item_count)
        self.processing_time.labels(stage=stage).observe(processing_time)
    
    def record_error(self, stage: str, error_type: str) -> None:
        """Record error metrics"""
        self.error_count.labels(stage=stage, error_type=error_type).inc()
    
    def update_queue_size(self, queue_name: str, size: int) -> None:
        """Update queue size metrics"""
        self.queue_size.labels(queue_name=queue_name).set(size)
    
    def update_system_metrics(self) -> None:
        """Update system metrics"""
        # CPU and memory
        self.cpu_utilization.set(psutil.cpu_percent())
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.used)
        
        # GPU metrics if available
        if self.config.enable_gpu_monitoring:
            self._update_gpu_metrics()
        
        # Ray object store
        if self.config.enable_ray_monitoring and ray.is_initialized():
            self._update_ray_metrics()
    
    def _update_gpu_metrics(self) -> None:
        """Update GPU metrics"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            for gpu in gpus:
                gpu_id = str(gpu.id)
                self.gpu_utilization.labels(gpu_id=gpu_id).set(gpu.load * 100)
                self.gpu_memory_used.labels(gpu_id=gpu_id).set(gpu.memoryUsed * 1024 * 1024)
                self.gpu_memory_total.labels(gpu_id=gpu_id).set(gpu.memoryTotal * 1024 * 1024)
                
        except ImportError:
            # GPUtil not available, skip GPU monitoring
            pass
        except Exception as e:
            logger.error(f"Error updating GPU metrics: {e}")
    
    def _update_ray_metrics(self) -> None:
        """Update Ray cluster metrics"""
        try:
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            # Object store memory usage
            object_store_spilled = ray._private.worker.global_worker.core_worker.get_object_spill_stats()
            total_spilled = sum(stat.total_bytes_spilled for stat in object_store_spilled.values())
            self.object_store_usage.set(total_spilled)
            
        except Exception as e:
            logger.error(f"Error updating Ray metrics: {e}")


class CheckpointManager:
    """Checkpoint management for fault tolerance"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint = 0
        
    def save_checkpoint(self, 
                       checkpoint_data: Dict[str, Any],
                       batch_id: str) -> str:
        """Save checkpoint data"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{batch_id}.json"
        
        checkpoint = {
            'batch_id': batch_id,
            'timestamp': time.time(),
            'data': checkpoint_data
        }
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            logger.info(f"Saved checkpoint: {checkpoint_file}")
            
            # Clean old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_file)
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time
        latest_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                checkpoint = json.load(f)
            
            logger.info(f"Loaded checkpoint: {latest_file}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def _cleanup_old_checkpoints(self, keep_last: int = 5) -> None:
        """Clean up old checkpoints"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        
        if len(checkpoint_files) <= keep_last:
            return
        
        # Sort by modification time and remove oldest
        checkpoint_files.sort(key=lambda f: f.stat().st_mtime)
        
        for old_file in checkpoint_files[:-keep_last]:
            try:
                old_file.unlink()
                logger.debug(f"Removed old checkpoint: {old_file}")
            except Exception as e:
                logger.error(f"Error removing old checkpoint {old_file}: {e}")


class FaultToleranceManager:
    """Fault tolerance and recovery management"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config)
        self.failed_tasks = defaultdict(list)
        self.retry_counts = defaultdict(int)
        self.max_retries = 3
        
    def record_failure(self, 
                      task_id: str,
                      error: str,
                      stage: str,
                      batch_data: Optional[Dict[str, Any]] = None) -> None:
        """Record a task failure"""
        failure_info = {
            'timestamp': time.time(),
            'error': error,
            'stage': stage,
            'batch_data': batch_data
        }
        
        self.failed_tasks[task_id].append(failure_info)
        logger.warning(f"Recorded failure for task {task_id} in stage {stage}: {error}")
    
    def can_retry(self, task_id: str) -> bool:
        """Check if a task can be retried"""
        return self.retry_counts[task_id] < self.max_retries
    
    def increment_retry(self, task_id: str) -> None:
        """Increment retry count for a task"""
        self.retry_counts[task_id] += 1
    
    def get_failed_tasks_for_retry(self, stage: str) -> List[Dict[str, Any]]:
        """Get failed tasks that can be retried"""
        retry_tasks = []
        
        for task_id, failures in self.failed_tasks.items():
            if (self.can_retry(task_id) and 
                failures and 
                failures[-1]['stage'] == stage):
                
                retry_tasks.append({
                    'task_id': task_id,
                    'batch_data': failures[-1]['batch_data'],
                    'retry_count': self.retry_counts[task_id]
                })
        
        return retry_tasks
    
    def clear_task_failures(self, task_id: str) -> None:
        """Clear failures for a successful task"""
        if task_id in self.failed_tasks:
            del self.failed_tasks[task_id]
        if task_id in self.retry_counts:
            del self.retry_counts[task_id]
    
    def save_progress_checkpoint(self, 
                                processed_batches: List[str],
                                current_batch_id: str) -> None:
        """Save progress checkpoint"""
        checkpoint_data = {
            'processed_batches': processed_batches,
            'current_batch_id': current_batch_id,
            'failed_tasks': dict(self.failed_tasks),
            'retry_counts': dict(self.retry_counts)
        }
        
        self.checkpoint_manager.save_checkpoint(checkpoint_data, current_batch_id)
    
    def load_progress_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load progress checkpoint"""
        checkpoint = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint:
            # Restore state
            self.failed_tasks = defaultdict(list, checkpoint['data'].get('failed_tasks', {}))
            self.retry_counts = defaultdict(int, checkpoint['data'].get('retry_counts', {}))
            
        return checkpoint


class MonitoringSystem:
    """Main monitoring system"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.fault_tolerance = FaultToleranceManager(config)
        self.running = False
        self.monitor_thread = None
        self.callbacks = defaultdict(list)
        
        # Performance tracking
        self.stage_performance = defaultdict(lambda: deque(maxlen=100))
        
    def start(self) -> None:
        """Start the monitoring system"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Monitoring system started")
        
    def stop(self) -> None:
        """Stop the monitoring system"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Monitoring system stopped")
        
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a monitoring callback"""
        self.callbacks[event].append(callback)
        
    def _trigger_callbacks(self, event: str, data: Dict[str, Any]) -> None:
        """Trigger callbacks for an event"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in monitoring callback: {e}")
    
    def record_stage_performance(self, 
                                stage: str,
                                batch_id: str,
                                processing_time: float,
                                item_count: int,
                                success: bool) -> None:
        """Record stage performance metrics"""
        # Update Prometheus metrics
        self.metrics_collector.record_batch_processed(stage, processing_time, item_count)
        
        if not success:
            self.metrics_collector.record_error(stage, 'processing_error')
        
        # Track performance history
        performance_data = {
            'batch_id': batch_id,
            'processing_time': processing_time,
            'item_count': item_count,
            'throughput': item_count / processing_time if processing_time > 0 else 0,
            'success': success,
            'timestamp': time.time()
        }
        
        self.stage_performance[stage].append(performance_data)
        
        # Trigger callbacks
        self._trigger_callbacks('stage_performance', {
            'stage': stage,
            'performance': performance_data
        })
    
    def get_stage_stats(self, stage: str) -> Dict[str, Any]:
        """Get performance statistics for a stage"""
        if stage not in self.stage_performance:
            return {}
        
        performances = list(self.stage_performance[stage])
        if not performances:
            return {}
        
        # Calculate statistics
        processing_times = [p['processing_time'] for p in performances]
        throughputs = [p['throughput'] for p in performances]
        success_count = sum(1 for p in performances if p['success'])
        
        return {
            'stage': stage,
            'total_batches': len(performances),
            'success_rate': success_count / len(performances),
            'avg_processing_time': sum(processing_times) / len(processing_times),
            'min_processing_time': min(processing_times),
            'max_processing_time': max(processing_times),
            'avg_throughput': sum(throughputs) / len(throughputs),
            'min_throughput': min(throughputs),
            'max_throughput': max(throughputs)
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        stats = {
            'timestamp': time.time(),
            'stages': {}
        }
        
        # Stage statistics
        for stage in self.stage_performance:
            stats['stages'][stage] = self.get_stage_stats(stage)
        
        # Fault tolerance statistics
        stats['fault_tolerance'] = {
            'failed_tasks': len(self.fault_tolerance.failed_tasks),
            'retry_counts': dict(self.fault_tolerance.retry_counts)
        }
        
        # System resources
        stats['system'] = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        return stats
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                # Update system metrics
                self.metrics_collector.update_system_metrics()
                
                # Periodic health check
                self._health_check()
                
                # Sleep until next update
                time.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.metrics_interval)
    
    def _health_check(self) -> None:
        """Perform health check"""
        # Check for stages with high error rates
        for stage in self.stage_performance:
            stats = self.get_stage_stats(stage)
            if stats.get('success_rate', 1.0) < 0.9:  # Less than 90% success rate
                logger.warning(f"Stage {stage} has low success rate: {stats['success_rate']:.2%}")
                
                # Trigger health alert callback
                self._trigger_callbacks('health_alert', {
                    'type': 'low_success_rate',
                    'stage': stage,
                    'success_rate': stats['success_rate']
                })
        
        # Check system resources
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            logger.warning(f"High memory usage: {memory_percent:.1f}%")
            self._trigger_callbacks('health_alert', {
                'type': 'high_memory_usage',
                'memory_percent': memory_percent
            })
    
    def export_metrics(self, output_path: str) -> None:
        """Export metrics to file"""
        try:
            stats = self.get_system_stats()
            
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
                
            logger.info(f"Exported metrics to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")


class AlertManager:
    """Alert management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules = config.get('alert_rules', [])
        self.alert_history = deque(maxlen=1000)
        
    def check_alerts(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check alert conditions and return triggered alerts"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            if self._evaluate_rule(rule, stats):
                alert = {
                    'rule_name': rule['name'],
                    'severity': rule.get('severity', 'warning'),
                    'message': rule.get('message', 'Alert triggered'),
                    'timestamp': time.time(),
                    'stats': stats
                }
                
                triggered_alerts.append(alert)
                self.alert_history.append(alert)
                
                logger.warning(f"Alert triggered: {rule['name']} - {alert['message']}")
        
        return triggered_alerts
    
    def _evaluate_rule(self, rule: Dict[str, Any], stats: Dict[str, Any]) -> bool:
        """Evaluate a single alert rule"""
        try:
            condition = rule['condition']
            
            # Simple condition evaluation (can be extended)
            if condition.get('type') == 'threshold':
                metric_path = condition['metric']
                threshold = condition['threshold']
                operator = condition.get('operator', '>')
                
                # Extract metric value from stats
                value = self._get_metric_value(stats, metric_path)
                
                if value is None:
                    return False
                
                # Evaluate condition
                if operator == '>':
                    return value > threshold
                elif operator == '<':
                    return value < threshold
                elif operator == '>=':
                    return value >= threshold
                elif operator == '<=':
                    return value <= threshold
                elif operator == '==':
                    return value == threshold
                
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating alert rule {rule.get('name', 'unknown')}: {e}")
            return False
    
    def _get_metric_value(self, stats: Dict[str, Any], path: str) -> Optional[float]:
        """Get metric value from stats using dot notation"""
        try:
            keys = path.split('.')
            value = stats
            
            for key in keys:
                value = value[key]
            
            return float(value)
            
        except (KeyError, TypeError, ValueError):
            return None