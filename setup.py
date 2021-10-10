import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gperc",
    version="0.1-beta",
    author="Yash Bonde",
    author_email="bonde.yash97@gmail.com",
    description="General purpose perceiver architectures!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yashbonde/general-perceivers",
    project_urls={
        "Bug Tracker": "https://github.com/yashbonde/general-perceivers/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause",
        "Operating System :: OS Independent",
    ],
    package_dir={"gperc": "gperc"},
    python_requires=">=3.6",
)
