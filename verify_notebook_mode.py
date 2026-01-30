import subprocess
import sys
import time

STEPS = [
    "step_1_1", "step_1_2", "step_1_3", "step_1_4",
    "step_2_1", "step_2_2", "step_2_3", "step_2_4", "step_2_5",
    "step_3_1", "step_3_2", "step_3_3",
    "step_4_1", "step_4_2", "step_4_3", "step_4_4",
    "step_5_1", "step_5_3"
]

def run_step(step_id):
    print(f"👉 Testing {step_id}...", end=" ", flush=True)
    start = time.time()
    result = subprocess.run(
        [sys.executable, "interactive_notebook.py", "--step", step_id],
        capture_output=True,
        text=True
    )
    duration = time.time() - start
    
    if result.returncode == 0:
        print(f"✅ PASS ({duration:.2f}s)")
        return True
    else:
        print(f"❌ FAIL")
        print("\n--- STDERR ---")
        print(result.stderr)
        return False

def main():
    print("🚀 Starting Notebook Pipeline Verification...")
    print(f"Steps to verify: {len(STEPS)}")
    
    passed = 0
    for step in STEPS:
        if run_step(step):
            passed += 1
        else:
            print("🛑 Aborting chain due to failure.")
            break
            
    print(f"\nSummary: {passed}/{len(STEPS)} steps passed.")
    
    if passed == len(STEPS):
        print("Integration Test Successful.")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
