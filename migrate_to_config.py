"""
Migration script to help transition from hardcoded parameters to configuration management.

This script analyzes existing code and provides recommendations for configuration migration.
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple


class ConfigMigrationAnalyzer:
    """Analyzes code for hardcoded values that should be moved to configuration."""
    
    def __init__(self):
        self.hardcoded_patterns = [
            # Numbers that might be hyperparameters
            (r'epochs\s*=\s*(\d+)', 'training.epochs'),
            (r'batch_size\s*=\s*(\d+)', 'training.batch_size'),
            (r'learning_rate\s*=\s*([\d.e-]+)', 'training.learning_rate'),
            (r'lr\s*=\s*([\d.e-]+)', 'training.learning_rate'),
            (r'weight_decay\s*=\s*([\d.e-]+)', 'training.weight_decay'),
            
            # Model parameters
            (r'd_model\s*=\s*(\d+)', 'model.backbone.d_model'),
            (r'n_layer\s*=\s*(\d+)', 'model.backbone.n_layer'),
            (r'nhead\s*=\s*(\d+)', 'model.backbone.nhead'),
            (r'dropout\s*=\s*([\d.]+)', 'model.backbone.dropout'),
            
            # Data parameters
            (r'sample_rate\s*=\s*(\d+)', 'data.sample_rate'),
            (r'num_workers\s*=\s*(\d+)', 'data.num_workers'),
            
            # Device parameters
            (r'cuda\s*=\s*(\d+)', 'device.cuda.device_id'),
            
            # Paths (quoted strings)
            (r'["\']([^"\']*(?:dataset|model|log|result)[^"\']*)["\']', 'paths.*'),
        ]
        
        self.string_patterns = [
            # Common paths
            (r'["\']([^"\']*\/data\/[^"\']*)["\']', 'paths.datasets_dir'),
            (r'["\']([^"\']*\/model[^"\']*)["\']', 'paths.model_dir'),
            (r'["\']([^"\']*\.pth)["\']', 'model.pretrained_weights.path'),
            (r'["\']([^"\']*pretrained[^"\']*)["\']', 'model.pretrained_weights.path'),
        ]
    
    def analyze_file(self, file_path: Path) -> Dict[str, List[Tuple[int, str, str]]]:
        """
        Analyze a Python file for hardcoded values.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'hardcoded_numbers': [],
            'hardcoded_strings': [],
            'suggestions': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                # Skip comments and empty lines
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith('#'):
                    continue
                
                # Check for hardcoded numbers
                for pattern, config_key in self.hardcoded_patterns:
                    matches = re.findall(pattern, line)
                    for match in matches:
                        results['hardcoded_numbers'].append((
                            line_num, 
                            match, 
                            config_key
                        ))
                
                # Check for hardcoded strings
                for pattern, config_key in self.string_patterns:
                    matches = re.findall(pattern, line)
                    for match in matches:
                        results['hardcoded_strings'].append((
                            line_num,
                            match,
                            config_key
                        ))
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
        
        return results
    
    def analyze_project(self, project_dir: Path) -> Dict[str, Any]:
        """
        Analyze entire project for hardcoded values.
        
        Args:
            project_dir: Path to the project directory
            
        Returns:
            Analysis results for all Python files
        """
        results = {}
        
        # Find all Python files
        python_files = list(project_dir.rglob("*.py"))
        
        for py_file in python_files:
            # Skip files in certain directories
            if any(part in str(py_file) for part in ['__pycache__', '.git', 'venv', 'env']):
                continue
            
            relative_path = py_file.relative_to(project_dir)
            results[str(relative_path)] = self.analyze_file(py_file)
        
        return results
    
    def generate_migration_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a migration report."""
        report = []
        report.append("=" * 80)
        report.append("CONFIGURATION MIGRATION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        total_hardcoded = 0
        
        for file_path, file_results in analysis_results.items():
            hardcoded_numbers = file_results['hardcoded_numbers']
            hardcoded_strings = file_results['hardcoded_strings']
            
            if hardcoded_numbers or hardcoded_strings:
                report.append(f"FILE: {file_path}")
                report.append("-" * len(f"FILE: {file_path}"))
                
                if hardcoded_numbers:
                    report.append("Hardcoded Numbers:")
                    for line_num, value, suggested_key in hardcoded_numbers:
                        report.append(f"  Line {line_num}: {value} → {suggested_key}")
                        total_hardcoded += 1
                
                if hardcoded_strings:
                    report.append("Hardcoded Strings:")
                    for line_num, value, suggested_key in hardcoded_strings:
                        report.append(f"  Line {line_num}: {value} → {suggested_key}")
                        total_hardcoded += 1
                
                report.append("")
        
        report.append("=" * 80)
        report.append("MIGRATION RECOMMENDATIONS")
        report.append("=" * 80)
        report.append("")
        report.append(f"Total hardcoded values found: {total_hardcoded}")
        report.append("")
        report.append("Recommended steps:")
        report.append("1. Review the configuration files in config/")
        report.append("2. Move hardcoded values to appropriate config sections")
        report.append("3. Update your code to use the ConfigManager:")
        report.append("   from config.utils import ConfigManager")
        report.append("   config = ConfigManager()")
        report.append("   value = config.get('training.learning_rate')")
        report.append("4. Test with different environments (development, production)")
        report.append("5. Use environment variables for deployment-specific values")
        report.append("")
        
        return "\n".join(report)


def create_sample_config_from_analysis(analysis_results: Dict[str, Any]) -> str:
    """Create a sample configuration based on analysis results."""
    config_values = {}
    
    for file_results in analysis_results.values():
        for _, value, config_key in file_results['hardcoded_numbers']:
            if config_key not in config_values:
                config_values[config_key] = value
    
    # Generate YAML structure
    yaml_lines = []
    yaml_lines.append("# Generated configuration based on code analysis")
    yaml_lines.append("# Review and adjust values as needed")
    yaml_lines.append("")
    
    # Group by sections
    sections = {}
    for key, value in config_values.items():
        if '.' in key:
            section = key.split('.')[0]
            if section not in sections:
                sections[section] = {}
            
            # Simple nested structure
            parts = key.split('.')
            current = sections[section]
            for part in parts[1:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
    
    # Convert to YAML-like format
    for section_name, section_data in sections.items():
        yaml_lines.append(f"{section_name}:")
        yaml_lines.extend(_dict_to_yaml(section_data, indent=2))
        yaml_lines.append("")
    
    return "\n".join(yaml_lines)


def _dict_to_yaml(data: Dict[str, Any], indent: int = 0) -> List[str]:
    """Convert dictionary to YAML-like format."""
    lines = []
    prefix = " " * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(_dict_to_yaml(value, indent + 2))
        else:
            lines.append(f"{prefix}{key}: {value}")
    
    return lines


def main():
    """Main migration analysis function."""
    parser = argparse.ArgumentParser(description='Analyze code for configuration migration')
    parser.add_argument('--project-dir', type=str, default='.',
                        help='Project directory to analyze')
    parser.add_argument('--output-file', type=str, default='migration_report.txt',
                        help='Output file for the migration report')
    parser.add_argument('--generate-config', action='store_true',
                        help='Generate sample configuration file')
    
    args = parser.parse_args()
    
    project_dir = Path(args.project_dir)
    
    print("Analyzing project for hardcoded values...")
    analyzer = ConfigMigrationAnalyzer()
    results = analyzer.analyze_project(project_dir)
    
    # Generate report
    report = analyzer.generate_migration_report(results)
    
    # Save report
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Migration analysis complete. Report saved to {args.output_file}")
    
    # Generate sample config if requested
    if args.generate_config:
        sample_config = create_sample_config_from_analysis(results)
        config_file = 'sample_config_from_analysis.yaml'
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(sample_config)
        
        print(f"Sample configuration saved to {config_file}")
    
    # Print summary
    total_files = len([f for f, r in results.items() if r['hardcoded_numbers'] or r['hardcoded_strings']])
    print(f"\nSummary:")
    print(f"- Analyzed {len(results)} Python files")
    print(f"- Found hardcoded values in {total_files} files")
    print(f"- See {args.output_file} for detailed recommendations")


if __name__ == '__main__':
    main()