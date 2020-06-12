import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE", "r") as fh:
    license_ = fh.readline().strip()

setuptools.setup(
    name="ml_utils-lluissalord", # Replace with your own username
    version="0.0.3",
    author="Lluis Salord Quetglas",
    author_email="l.salord.quetglas@gmail.com",
    description="Library for functions useful for daily usage on Data Science and Data Analytics fields.",
    license=license_,
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
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'tqdm',
    ]
)