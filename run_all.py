#run_all.py
import subprocess
import sys

def run(script):
    print(f"\n🚀 Running {script} ...\n")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"❌ Error while running {script}")
        sys.exit(1)
 
print("🌱 Eco-AI Pipeline Starting...\n")

run("src/baseline_train.py")
run("src/eco_train.py")

print("\n📊 Launching Dashboard...\n")
subprocess.run([sys.executable, "src/dashboard.py"])