from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
    "torch"
]

setup(
    name="titans-unofficial",
    version="1.1.0",
    author="Shehryar Sohail",
    author_email="hafizshehryar@gmail.com",
    description="Unofficial PyTorch implementation of Titans: Learning to Memorize at Test Time",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shehryar718/titans-unofficial",
    packages=find_packages(),
    package_data={
        "titans": ["models/*.py", "utils/*.py"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.23.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "examples": [
            "numpy",
            "transformers",
            "einops"
        ],
    },
    zip_safe=False,
)
