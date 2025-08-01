#!/usr/bin/env python3
"""
NIS Agent Toolkit - Setup and Installation
Agent-level development tools for NIS Protocol
"""

from setuptools import setup, find_packages

setup(
    name="nis-agent-toolkit",
    version="1.0.0",
    description="Agent-level development toolkit for NIS Protocol intelligent agents",
    long_description=open("README.md").read() if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Organica AI Solutions",
    author_email="dev@organica-ai.com",
    url="https://github.com/organica-ai/nis-agent-toolkit",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "click>=8.0.0",
        "pydantic>=2.0.0", 
        "pyyaml>=6.0",
        "rich>=13.0.0",
        "asyncio-toolkit>=0.5.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "vision": [
            "opencv-python>=4.5.0",
            "pillow>=9.0.0",
        ],
        "ml": [
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "nis-agent=cli.main:main",
            "nis-agent-create=cli.create:create_agent",
            "nis-agent-test=cli.test:test_agent",
            "nis-agent-simulate=cli.simulate:simulate_agent",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="nis protocol ai agents toolkit development reasoning",
    project_urls={
        "Bug Reports": "https://github.com/organica-ai/nis-agent-toolkit/issues",
        "Source": "https://github.com/organica-ai/nis-agent-toolkit",
        "Documentation": "https://docs.organica-ai.com/nis-agent-toolkit",
    },
)
