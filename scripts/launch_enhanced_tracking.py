#!/usr/bin/env python3
"""
Enhanced Tracking System Launcher
Main entry point for the enhanced face tracking system
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Launch the enhanced tracking system."""
    print("🎯 ENHANCED TRACKING SYSTEM LAUNCHER")
    print("=" * 60)
    print("Choose an option:")
    print("1. Start Enhanced Tracking (Live Camera)")
    print("2. Run Tracking Demo (Simulated)")
    print("3. Run System Tests")
    print("4. Exit")
    print("=" * 60)
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                print("\n🚀 Starting Enhanced Tracking with Live Camera...")
                script_path = Path("../scripts/start_enhanced_tracking.py").resolve()
                if script_path.exists():
                    subprocess.run([sys.executable, str(script_path)])
                else:
                    print("❌ Enhanced tracking script not found!")
                break

            elif choice == '2':
                print("\n🎮 Starting Tracking Demo...")
                script_path = Path("../demos/test_tracking_demo.py").resolve()
                if script_path.exists():
                    subprocess.run([sys.executable, str(script_path)])
                else:
                    print("❌ Demo script not found!")
                break

            elif choice == '3':
                print("\n🧪 Running System Tests...")
                script_path = Path("../tests/test_enhanced_tracking.py").resolve()
                if script_path.exists():
                    subprocess.run([sys.executable, str(script_path)])
                else:
                    print("❌ Test script not found!")
                break
                
            elif choice == '4':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

if __name__ == "__main__":
    main()
