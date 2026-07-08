import os
import zipfile

def zip_professional():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    zip_name = os.path.join(src_dir, "CAPL_professional.zip")
    
    # Specific python files to include from src/
    src_files_to_include = {
        'parser.py',
        'solution_representation.py',
        'cost_calculator.py',
        'constraint_checker.py',
        'baseline.py',
        'baseline_solver.py',
        'chromosome.py',
        'population.py',
        'crossover.py',
        'mutation.py',
        'selection.py',
        'repair.py',
        'fitness.py',
        'genetic_algorithm.py',
        'surrogate_model.py',
        'feature_engineering.py',
        'evaluation_metrics.py',
        'dataset_generator.py',
        'training_pipeline.py',
        'active_learning.py',
        'hybrid_ga.py',
        'benchmark_large.py',
        'ga_solver.py',
        'benchmark_cflp.py',
        'benchmark_statistical.py',
        'verify_ga_optimal.py',
        'verify_cap_stats.py',
        'verify_parser.py',
        'verify_phase2.py'
    }
    
    # Specific docs/outputs files to include
    docs_files_to_include = {
        'cap41_ga_convergence.png',
        'cap41_hybrid_convergence.png',
        'hybrid_ga_comparison.png',
        'large_benchmark_results.csv',
        'uflp_benchmark_results.csv',
        'statistical_benchmark_results.csv',
        'statistical_benchmark_results.png',
        'computational_table.png',
        'CFLP_Complete_Project_Guide.md'
    }
    
    # Root files to include
    root_files_to_include = {
        'README.md',
        'requirements.txt',
        'project_progress_report.md',
        'computation_results.png',
        'zip.py' # include zip script itself
    }
    
    print(f"Creating professional zip archive: {zip_name}")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add root files
        for file in root_files_to_include:
            full_path = os.path.join(src_dir, file)
            if os.path.exists(full_path):
                zipf.write(full_path, file)
                print(f"  Added: {file}")
            else:
                print(f"  [Warning] Root file not found: {file}")
                
        # Add source files
        for file in src_files_to_include:
            full_path = os.path.join(src_dir, 'src', file)
            if os.path.exists(full_path):
                zipf.write(full_path, os.path.join('src', file))
                print(f"  Added: src/{file}")
            else:
                print(f"  [Warning] Source file not found: {file}")
                
        # Add outputs in docs/
        for file in docs_files_to_include:
            full_path = os.path.join(src_dir, 'docs', file)
            if os.path.exists(full_path):
                zipf.write(full_path, os.path.join('docs', file))
                print(f"  Added: docs/{file}")
            else:
                print(f"  [Warning] Output file not found: {file}")
                
        # Add data (raw and processed)
        data_dir = os.path.join(src_dir, 'data')
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, src_dir)
                    zipf.write(full_path, rel_path)
            print("  Added: All files in data/ directory")
            
    print("\nProfessional zip archive created successfully!")

if __name__ == '__main__':
    zip_professional()