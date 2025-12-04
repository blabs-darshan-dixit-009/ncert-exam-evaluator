#!/usr/bin/env python3
import requests
import time
import sys

BASE_URL = "http://localhost:8001"

def monitor():
    print("🔍 Monitoring Training Progress")
    print("=" * 60)
    
    while True:
        try:
            # Get progress
            response = requests.get(f"{BASE_URL}/api/training/progress", timeout=2)
            data = response.json()
            
            status = data.get("status", "unknown")
            
            if status == "no_training":
                print("\n❌ No training in progress")
                break
            
            # Display progress
            print(f"\r🔄 {data.get('stage', 'Unknown')} | "
                  f"Epoch {data.get('current_epoch', 0)}/{data.get('total_epochs', 0)} | "
                  f"Step {data.get('current_step', 0)}/{data.get('total_steps', 0)} | "
                  f"Loss: {data.get('training_loss', 'N/A')} | "
                  f"{data.get('progress_percentage', 0)}%", 
                  end='', flush=True)
            
            if status in ["completed", "failed"]:
                print(f"\n\n{'✅' if status == 'completed' else '❌'} Training {status}!")
                if status == "completed":
                    print(f"Final Loss: {data.get('training_loss')}")
                break
            
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            
        time.sleep(2)  # Update every 2 seconds

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped")
        sys.exit(0)