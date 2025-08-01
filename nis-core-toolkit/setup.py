#!/usr/bin/env python3
"""
NIS Core Toolkit (NDT) - System-level developer toolkit for NIS Protocol
Built with engineering integrity - no hype, just working tools
"""

from setuptools import setup, find_packages

setup(
    name="nis-core-toolkit",
    version="1.0.0",
    description="System-level developer toolkit for NIS Protocol multi-agent systems",
    author="Organica AI Solutions",
    python_requires=">=3.9",
    install_requires=[
        "click>=8.0.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "fastapi>=0.100.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "nis=cli.main:main",
            "nis-init=cli.init:main",
            "nis-create=cli.create:main",
            "nis-validate=cli.validate:main",
            "nis-deploy=cli.deploy:main",
        ],
    },
)
