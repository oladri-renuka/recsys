from pathlib import Path
import json
import torch

print("="*80)
print("VERIFYING SASREC LOCAL SETUP")
print("="*80)

checks = []

# Check model file
print("\n1. Checking model file...")
model_path = Path("results/best_model.pt")
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024**2)
    print(f"   ✓ best_model.pt found ({size_mb:.1f} MB)")
    checks.append(True)
else:
    print(f"   ✗ best_model.pt NOT found")
    checks.append(False)

# Check JSON files
print("\n2. Checking result files...")
json_files = [
    'training_log.json',
    'ablation_results.json',
    'segment_analysis.json',
    'benchmark_results.json'
]

for json_file in json_files:
    path = Path(f"results/{json_file}")
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"   ✓ {json_file} ({len(str(data))} bytes)")
        checks.append(True)
    else:
        print(f"   ✗ {json_file} NOT found")
        checks.append(False)

# Check Python files
print("\n3. Checking code files...")
code_files = ['model.py', 'data.py', 'dataset.py', 'inference.py', 'app.py', 'train.py']

for code_file in code_files:
    path = Path(code_file)
    if path.exists():
        lines = len(path.read_text().split('\n'))
        print(f"   ✓ {code_file} ({lines} lines)")
        checks.append(True)
    else:
        print(f"   ✗ {code_file} NOT found")
        checks.append(False)

# Check PyTorch
print("\n4. Checking PyTorch...")
try:
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    checks.append(True)
except:
    print(f"   ✗ PyTorch not installed")
    checks.append(False)

# Test inference
print("\n5. Testing inference...")
try:
    from inference import SASRecInference
    recommender = SASRecInference(device='cpu')
    recs = recommender.recommend([5, 10, 15, 20], n_recommendations=5)
    print(f"   ✓ Inference working (got {len(recs)} recommendations)")
    checks.append(True)
except Exception as e:
    print(f"   ✗ Inference failed: {e}")
    checks.append(False)

# Summary
print("\n" + "="*80)
if all(checks):
    print(" ALL CHECKS PASSED - Setup is ready!")
else:
    print(f" {sum(checks)}/{len(checks)} checks passed")

print("="*80)