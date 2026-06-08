import subprocess
import sys
import threading

def run_training():
    print("\n🚀 Running baseline_train.py ...\n")
    subprocess.run([sys.executable, "src/baseline_train.py"])
    print("\n🚀 Running eco_train.py ...\n")
    subprocess.run([sys.executable, "src/eco_train.py"])
    print("\n✅ Training complete.\n")

# Run training in background so dashboard starts immediately
t = threading.Thread(target=run_training, daemon=True)
t.start()

print("\n📊 Launching Dashboard...\n")
subprocess.run([sys.executable, "src/dashboard.py"])
