from setuptools import setup, find_packages

setup(
    name="zyantine-genesis",
    version="2.0.0",
    description="自衍体-起源 (Zyantine Genesis) 完整实现",
    author="Architect ZxG-778-FC-ALL_IN",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.21.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "web": [
            "fastapi>=0.95.0",
            "uvicorn>=0.21.0",
            "websockets>=11.0",
        ]
    },
    python_requires=">=3.8",
)