from setuptools import setup, find_packages

setup(
    name="mlforgex",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn"
    ],
    entry_points={
        "console_scripts": [
            "mlforge-train=mlforge.train:main",
            "mlforge-predict=mlforge.predict:main"
        ]
    },
    author="Priyanshu Mathur",
    author_email="mathurpriyanshu2006@gmail.com",
    description="MLForge: Train and evaluate ML models easily",
    url="https://github.com/yourusername/mlforge",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
