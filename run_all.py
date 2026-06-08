import subprocess
import sys
import threading
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_training():
    print("\n🚀 Running baseline_train.py ...\n")
    subprocess.run([sys.executable, "src/baseline_train.py"], cwd=BASE_DIR)
    print("\n🚀 Running eco_train.py ...\n")
    subprocess.run([sys.executable, "src/eco_train.py"], cwd=BASE_DIR)
    print("\n✅ Training complete.\n")

t = threading.Thread(target=run_training, daemon=True)
t.start()

print("\n📊 Launching Dashboard...\n")
subprocess.run([sys.executable, "src/dashboard.py"], cwd=BASE_DIR)
