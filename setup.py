#!/usr/bin/env python3
"""
Setup script for PIPER - AI Classroom Analyzer
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="piper-ai-classroom-analyzer",
    version="1.0.0",
    author="Subhasis Jena",
    author_email="subhasisjena1643@gmail.com",
    description="AI-powered classroom monitoring system with real-time face tracking and engagement analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/subhasisjena1643/classanalyzer",
    project_urls={
        "Bug Tracker": "https://github.com/subhasisjena1643/classanalyzer/issues",
        "Documentation": "https://github.com/subhasisjena1643/classanalyzer#readme",
        "Source Code": "https://github.com/subhasisjena1643/classanalyzer",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Multimedia :: Video :: Capture",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "piper-main=main_app:main",
            "piper-full=run_app:main",
            "piper-tracking=scripts.start_enhanced_tracking:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    keywords=[
        "artificial-intelligence",
        "computer-vision",
        "face-recognition",
        "education-technology",
        "classroom-monitoring",
        "engagement-analysis",
        "real-time-tracking",
        "reinforcement-learning",
        "deep-learning",
        "opencv",
        "mediapipe",
        "pytorch",
        "tensorflow",
    ],
    zip_safe=False,
)
