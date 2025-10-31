#!/usr/bin/env python3
"""
Quick start script for the RL Visual Input project
"""

import os
import sys
import subprocess
import importlib

def check_requirements():
    """Check if required packages are installed"""
    # Map pip package names to the modules we need to import
    required_packages = [
        ('torch', 'torch'),
        ('stable-baselines3', 'stable_baselines3'),
        ('gym', 'gym'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('opencv-python', 'cv2'),
        ('Pillow', 'PIL'),
    ]
    
    missing_packages = []
    for pip_name, module_name in required_packages:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("All required packages are installed")
        return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data/raw/cartpole',
        'data/processed',
        'data/generated',
        'results/plots',
        'results/logs',
        'results/models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure created")

def run_demo():
    """Run a quick demo of the first experiment"""
    print("\nRunning quick demo (Experiment 1: Native State RL)")
    print("This will train for 10,000 timesteps...")
    
    try:
        result = subprocess.run([
            sys.executable, 'experiments/exp1_native/run_experiment.py'
        ], cwd=os.getcwd(), timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("Demo completed successfully!")
            print("Check results/plots/ for training curves")
        else:
            print("Demo failed. Check the error messages above.")
            
    except subprocess.TimeoutExpired:
        print("Demo timed out. This is normal for longer training runs.")
    except Exception as e:
        print(f"Demo failed with error: {e}")

def main():
    print("RL Visual Input Project - Quick Start")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Setup directories
    setup_directories()
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Run quick demo (Experiment 1)")
    print("2. Run all experiments")
    print("3. Just setup (no running)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        run_demo()
    elif choice == '2':
        print("\nRunning all experiments...")
        subprocess.run([sys.executable, 'run_all_experiments.py'])
    elif choice == '3':
        print("  Setup complete! You can now run experiments manually.")
        print("\nTo run experiments:")
        print("  python run_all_experiments.py")
        print("  python experiments/exp1_native/run_experiment.py")
        print("  python analyze_results.py")
    elif choice == '4':
        print(" Goodbye!")
    else:
        print(" Invalid choice. Please run the script again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
