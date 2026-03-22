from setuptools import setup, find_packages

setup(
    name="gym_rotor_isaac",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "plum-dispatch",
    ],
)
