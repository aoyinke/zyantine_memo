"""
项目安装配置文件
"""
from setuptools import setup, find_packages

setup(
    name="zyantine_architecture",
    version="1.0.0",
    description="自衍体AI系统",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "nltk>=3.8.0",
        "scikit-learn>=1.0.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
)