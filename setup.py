from setuptools import setup, find_packages

def get_long_description():
    """Read long description from README"""
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
    return long_description

setup(
    name="fedbench",
    version="0.1.1",  # Update with your version
    author="Mohammed Nechba",
    author_email="mohammednechba@gmail.com",
    description="A Federated Learning Benchmarking Framework",
     long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/NechbaMohammed/FLBenchmark",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "flwr[simulation]==1.16.0",
        "omegaconf>=2.1.1",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "datasets>=1.18.0",  # HuggingFace datasets
        "hydra-core>=1.1.1",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "isort>=5.0.0",
        ],
    },
)
