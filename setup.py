from setuptools import setup, find_packages

with open("src/img2img/README.md", "r") as f:
    long_description = f.read()

setup(
    name="img2img",
    version="0.0.1",
    description="An image to image generator",
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    long_description=long_description,
    install_requires=[
        "torch>=2.0.1",
    ],
    python_requires=">=3.10"
)
