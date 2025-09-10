#!/usr/bin/env python3
"""
CBraMod Deployment Constraints Analysis
======================================

Analyzes model size, latency, and memory footprint compatibility with near-device inference
on IDUN-like hardware. Measures parameter count, memory usage in fp32/mixed precision,
per-epoch inference latency, and provides deployment feasibility assessment.

Research Question: Are model size, latency, and memory footprint compatible with near-device 
inference on IDUN-like hardware, and which compression or quantization strategies are promising?

Usage:
    python deployment_analysis.py --model_path saved_models/pretrained/pretrained_weights.pth
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import psutil
import argparse
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import gc
from contextlib import contextmanager

# Import CBraMod model
from cbramod.models.cbramod import CBraMod
from cbramod.models.model_for_idun import Model as IDUNModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentAnalyzer:
    """Comprehensive deployment constraints analysis for CBraMod."""
    
    def __init__(self, model_path: str, output_dir: str = "deployment_analysis"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # IDUN hardware constraints (based on typical edge device specs)
        self.idun_constraints = {
            'max_memory_mb': 512,      # 512MB typical for edge devices
            'max_latency_ms': 100,     # Sub-100ms requirement
            'max_power_mw': 1000,      # 1W power budget typical
            'precision_support': ['fp32', 'fp16', 'int8']  # Common precisions
        }
        
    @contextmanager
    def memory_monitor(self):
        """Context manager to monitor memory usage."""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        initial_memory = self._get_memory_usage()
        yield
        
        peak_memory = self._get_memory_usage()
        memory_used = peak_memory - initial_memory
        self.results['memory_used_mb'] = memory_used
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
    
    def load_model(self) -> nn.Module:
        """Load CBraMod model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load state dict first
        state_dict = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Determine model type based on keys
        if any('backbone.' in key for key in state_dict.keys()):
            logger.info("Detected IDUN Model format")
            # Detect number of classes from classifier weights
            num_classes = 4  # Default
            if 'classifier.weight' in state_dict:
                num_classes = state_dict['classifier.weight'].shape[0]
                logger.info(f"Detected {num_classes} classes from classifier weights")
            
            # Create a simple param object for IDUN Model
            class SimpleParam:
                def __init__(self):
                    self.use_pretrained_weights = False
                    self.num_of_classes = num_classes
                    self.head_type = 'simple'
                    self.cuda = 0
            
            param = SimpleParam()
            model = IDUNModel(param)
            
            # Load with strict=False to ignore missing/extra keys
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys (will use random init): {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys (ignored): {unexpected_keys[:5]}...")
        else:
            logger.info("Detected CBraMod format")
            model = CBraMod()
            # Load with strict=False to handle any key mismatches
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys (will use random init): {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys (ignored): {unexpected_keys[:5]}...")
        
        model.to(self.device)
        model.eval()
        
        logger.info(f"Successfully loaded model on {self.device}")
        return model
    
    def analyze_model_size(self, model: nn.Module) -> Dict:
        """Analyze model architecture and parameter count."""
        logger.info("Analyzing model size and architecture...")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Parameter breakdown by layer type
        param_breakdown = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                layer_type = type(module).__name__
                if layer_type not in param_breakdown:
                    param_breakdown[layer_type] = 0
                param_breakdown[layer_type] += module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    param_breakdown[layer_type] += module.bias.numel()
        
        # Model size in different precisions
        fp32_size_mb = total_params * 4 / 1024 / 1024  # 4 bytes per fp32
        fp16_size_mb = total_params * 2 / 1024 / 1024  # 2 bytes per fp16
        int8_size_mb = total_params * 1 / 1024 / 1024  # 1 byte per int8
        
        size_analysis = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_breakdown': param_breakdown,
            'model_size_fp32_mb': fp32_size_mb,
            'model_size_fp16_mb': fp16_size_mb,
            'model_size_int8_mb': int8_size_mb,
            'compression_ratio_fp16': fp32_size_mb / fp16_size_mb,
            'compression_ratio_int8': fp32_size_mb / int8_size_mb
        }
        
        return size_analysis
    
    def create_synthetic_batch(self, batch_size: int = 1) -> torch.Tensor:
        """Create synthetic input batch for inference testing."""
        # IDUN Model expects (batch_size, channels, sequence_length, epoch_size)
        # Typical IDUN EEG: 30s epochs, 200Hz sampling = 6000 samples per epoch
        channels = 1          # Single EEG channel
        sequence_length = 30  # 30 patches per epoch (30s / 1s patches)  
        epoch_size = 200      # Patch embedding dimension
        
        return torch.randn(batch_size, channels, sequence_length, epoch_size, device=self.device)
    
    def measure_inference_latency(self, model: nn.Module, num_warmup: int = 10, num_trials: int = 100) -> Dict:
        """Measure inference latency with multiple trials."""
        logger.info("Measuring inference latency...")
        
        # Create test batch
        test_input = self.create_synthetic_batch(batch_size=1)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(test_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Timed runs
        latencies = []
        with torch.no_grad():
            for _ in range(num_trials):
                start_time = time.perf_counter()
                _ = model(test_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
        
        latency_stats = {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'num_trials': num_trials
        }
        
        return latency_stats
    
    def measure_memory_footprint(self, model: nn.Module) -> Dict:
        """Measure memory footprint during inference."""
        logger.info("Measuring memory footprint...")
        
        test_input = self.create_synthetic_batch(batch_size=1)
        
        # Measure memory for different precisions
        memory_results = {}
        
        # FP32 (default)
        model.float()
        test_input = test_input.float()
        with self.memory_monitor():
            with torch.no_grad():
                _ = model(test_input)
        memory_results['fp32_memory_mb'] = self.results['memory_used_mb']
        
        # FP16 if supported
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            try:
                model.half()
                test_input = test_input.half()
                with self.memory_monitor():
                    with torch.no_grad():
                        _ = model(test_input)
                memory_results['fp16_memory_mb'] = self.results['memory_used_mb']
                
                # Reset to fp32
                model.float()
            except Exception as e:
                logger.warning(f"FP16 testing failed: {e}")
                memory_results['fp16_memory_mb'] = None
        else:
            memory_results['fp16_memory_mb'] = None
        
        return memory_results
    
    def assess_deployment_feasibility(self, analysis_results: Dict) -> Dict:
        """Assess deployment feasibility against IDUN constraints."""
        logger.info("Assessing deployment feasibility...")
        
        feasibility = {
            'memory_feasible': {
                'fp32': analysis_results['size_analysis']['model_size_fp32_mb'] < self.idun_constraints['max_memory_mb'],
                'fp16': analysis_results['size_analysis']['model_size_fp16_mb'] < self.idun_constraints['max_memory_mb'],
                'int8': analysis_results['size_analysis']['model_size_int8_mb'] < self.idun_constraints['max_memory_mb']
            },
            'latency_feasible': analysis_results['latency_analysis']['mean_latency_ms'] < self.idun_constraints['max_latency_ms'],
            'recommended_precision': 'fp32',  # Default
            'compression_needed': False,
            'deployment_score': 0.0  # 0-1 score
        }
        
        # Determine recommended precision
        if feasibility['memory_feasible']['fp32'] and feasibility['latency_feasible']:
            feasibility['recommended_precision'] = 'fp32'
        elif feasibility['memory_feasible']['fp16']:
            feasibility['recommended_precision'] = 'fp16'
            feasibility['compression_needed'] = True
        else:
            feasibility['recommended_precision'] = 'int8'
            feasibility['compression_needed'] = True
        
        # Calculate deployment score (higher is better)
        memory_score = 1.0 if feasibility['memory_feasible'][feasibility['recommended_precision']] else 0.5
        latency_score = min(1.0, self.idun_constraints['max_latency_ms'] / analysis_results['latency_analysis']['mean_latency_ms'])
        feasibility['deployment_score'] = (memory_score + latency_score) / 2
        
        return feasibility
    
    def generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate deployment recommendations."""
        recommendations = []
        
        size_analysis = analysis_results['size_analysis']
        latency_analysis = analysis_results['latency_analysis']
        feasibility = analysis_results['feasibility_assessment']
        
        # Memory recommendations
        if not feasibility['memory_feasible']['fp32']:
            recommendations.append(
                f"Model too large for IDUN hardware in FP32 ({size_analysis['model_size_fp32_mb']:.1f}MB > {self.idun_constraints['max_memory_mb']}MB). "
                f"Use FP16 (saves {(1 - 1/size_analysis['compression_ratio_fp16'])*100:.1f}%) or quantization."
            )
        
        # Latency recommendations  
        if not feasibility['latency_feasible']:
            recommendations.append(
                f"Latency too high ({latency_analysis['mean_latency_ms']:.1f}ms > {self.idun_constraints['max_latency_ms']}ms). "
                "Consider model pruning, layer fusion, or hardware acceleration."
            )
        
        # Quantization recommendations
        int8_savings = (1 - 1/size_analysis['compression_ratio_int8']) * 100
        if int8_savings > 50:
            recommendations.append(
                f"INT8 quantization could save {int8_savings:.1f}% memory. "
                "Test accuracy retention with post-training quantization."
            )
        
        # Architecture recommendations
        transformer_params = size_analysis['parameter_breakdown'].get('MultiheadAttention', 0)
        if transformer_params > size_analysis['total_parameters'] * 0.5:
            recommendations.append(
                "Transformer layers dominate parameter count. Consider attention head reduction or distillation."
            )
        
        if not recommendations:
            recommendations.append("Model meets IDUN deployment constraints in current form.")
        
        return recommendations
    
    def run_complete_analysis(self) -> Dict:
        """Run complete deployment analysis."""
        logger.info("Starting comprehensive deployment analysis...")
        
        # Load model
        model = self.load_model()
        
        # Run analyses
        size_analysis = self.analyze_model_size(model)
        latency_analysis = self.measure_inference_latency(model)
        memory_analysis = self.measure_memory_footprint(model)
        
        # Combine results
        analysis_results = {
            'model_path': str(self.model_path),
            'device': str(self.device),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'idun_constraints': self.idun_constraints,
            'size_analysis': size_analysis,
            'latency_analysis': latency_analysis,
            'memory_analysis': memory_analysis
        }
        
        # Assess feasibility
        feasibility = self.assess_deployment_feasibility(analysis_results)
        analysis_results['feasibility_assessment'] = feasibility
        
        # Generate recommendations
        recommendations = self.generate_recommendations(analysis_results)
        analysis_results['recommendations'] = recommendations
        
        return analysis_results
    
    def save_results(self, results: Dict):
        """Save results in multiple formats."""
        
        # Save detailed JSON
        json_path = self.output_dir / 'deployment_analysis.json'
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2)
        
        # Save CSV summary
        csv_data = self._create_csv_summary(results)
        csv_path = self.output_dir / 'deployment_summary.csv'
        pd.DataFrame([csv_data]).to_csv(csv_path, index=False)
        
        # Save recommendations text
        rec_path = self.output_dir / 'deployment_recommendations.txt'
        with open(rec_path, 'w') as f:
            f.write("CBraMod Deployment Analysis Recommendations\n")
            f.write("=" * 50 + "\n\n")
            for i, rec in enumerate(results['recommendations'], 1):
                f.write(f"{i}. {rec}\n\n")
        
        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"  - Detailed analysis: {json_path}")
        logger.info(f"  - Summary CSV: {csv_path}")
        logger.info(f"  - Recommendations: {rec_path}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        else:
            return obj
    
    def _create_csv_summary(self, results: Dict) -> Dict:
        """Create summary data for CSV export."""
        size = results['size_analysis']
        latency = results['latency_analysis']
        feasibility = results['feasibility_assessment']
        
        return {
            'model_path': results['model_path'],
            'total_parameters': size['total_parameters'],
            'model_size_fp32_mb': size['model_size_fp32_mb'],
            'model_size_fp16_mb': size['model_size_fp16_mb'],
            'model_size_int8_mb': size['model_size_int8_mb'],
            'mean_latency_ms': latency['mean_latency_ms'],
            'p95_latency_ms': latency['p95_latency_ms'],
            'memory_feasible_fp32': feasibility['memory_feasible']['fp32'],
            'memory_feasible_fp16': feasibility['memory_feasible']['fp16'],
            'latency_feasible': feasibility['latency_feasible'],
            'recommended_precision': feasibility['recommended_precision'],
            'deployment_score': feasibility['deployment_score'],
            'compression_needed': feasibility['compression_needed']
        }
    
    def print_summary(self, results: Dict):
        """Print analysis summary to console."""
        print("\n" + "="*70)
        print("CBraMod DEPLOYMENT ANALYSIS SUMMARY")
        print("="*70)
        
        size = results['size_analysis']
        latency = results['latency_analysis']
        feasibility = results['feasibility_assessment']
        
        print(f"\n📊 MODEL SIZE:")
        print(f"  Parameters: {size['total_parameters']:,}")
        print(f"  FP32: {size['model_size_fp32_mb']:.1f}MB")
        print(f"  FP16: {size['model_size_fp16_mb']:.1f}MB ({size['compression_ratio_fp16']:.1f}x compression)")
        print(f"  INT8: {size['model_size_int8_mb']:.1f}MB ({size['compression_ratio_int8']:.1f}x compression)")
        
        print(f"\n⚡ INFERENCE LATENCY:")
        print(f"  Mean: {latency['mean_latency_ms']:.2f}ms")
        print(f"  P95:  {latency['p95_latency_ms']:.2f}ms")
        print(f"  P99:  {latency['p99_latency_ms']:.2f}ms")
        
        print(f"\n🎯 IDUN COMPATIBILITY:")
        print(f"  Memory (FP32): {'✅' if feasibility['memory_feasible']['fp32'] else '❌'}")
        print(f"  Memory (FP16): {'✅' if feasibility['memory_feasible']['fp16'] else '❌'}")
        print(f"  Latency (<100ms): {'✅' if feasibility['latency_feasible'] else '❌'}")
        print(f"  Deployment Score: {feasibility['deployment_score']:.2f}/1.0")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='CBraMod Deployment Constraints Analysis')
    parser.add_argument('--model_path', 
                       default='saved_models/pretrained/pretrained_weights.pth',
                       help='Path to trained CBraMod model')
    parser.add_argument('--output_dir', 
                       default='deployment_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = DeploymentAnalyzer(args.model_path, args.output_dir)
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        # Save and display results
        analyzer.save_results(results)
        analyzer.print_summary(results)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())