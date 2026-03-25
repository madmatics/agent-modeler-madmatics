from setuptools import setup, find_packages

setup(
    name="madmatics",
    version="0.13.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "arviz>=0.16",
        "matplotlib>=3.7",
        "mlflow>=2.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "pydantic>=2.0",
        "pymc>=5.0",
        "pyprojroot",
        "pytensor>=2.0",
        "scipy>=1.10",
        "seaborn>=0.12",
        "scikit-learn>=1.2",
        "xarray>=2023.0",
        "xarray-einstats>=0.5",
    ],
    package_data={
        "madmatics": ["py.typed"],
    },
)
