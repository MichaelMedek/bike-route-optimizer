import pathlib

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
version_file = HERE / "version.txt"

with open(version_file, "r", encoding="utf-8") as fh:
    version = fh.readlines()[-1].strip()

with open(HERE / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(HERE / "requirements.txt", "r", encoding="utf-8") as fh:
    # Keep only real dependency specs: drop blanks, comments, and pip option lines
    # (-e, --index-url, -r, …) which are not install_requires entries.
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith(("#", "-"))]

setup(
    name="bike-route-optimizer",
    version=version,
    author="Michael Medek",
    author_email="michimedi@gmail.com",
    description="Eco- & surface-optimized bicycle route planner (flat routing, asphalt-preferring), CLI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichaelMedek/bike-route-optimizer",
    packages=find_packages(include=["bike_router", "bike_router.*"]),
    py_modules=["bike_route"],
    install_requires=requirements,
    entry_points={"console_scripts": ["bike-route=bike_route:main"]},
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    include_package_data=True,
    zip_safe=False,
)
