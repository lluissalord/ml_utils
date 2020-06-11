import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml_utils-lluissalord", # Replace with your own username
    version="0.0.1",
    author="Lluis Salord Quetglas",
    author_email="l.salord.quetglas@gmail.com",
    description="Library for functions useful for daily usage on Data Science and Data Analytics fields.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lluissalord/ml_utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)