from setuptools import setup, find_packages

setup(
    name="rl-visual-input",
    version="0.1.0",
    description="Reinforcement Learning with Visual Inputs Research Project",
    author="Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=1.11.0",
        "stable-baselines3>=1.8.0",
        "gym>=0.25.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.3.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
    ],
    python_requires=">=3.8",
)
