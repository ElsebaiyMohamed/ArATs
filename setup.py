import setuptools

# with open("README.md", "r", encoding = "utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name = "aang",
    version = "0.0.1",
    author = "seba3y",
    author_email = "sebaeymohamed43@gmail.com",
    description = "speech translation",
    long_description = 'long_description',
    long_description_content_type = "text/markdown",
    package_dir={"": '.aang'},
    
    
    
    
    url = "package URL",
    project_urls = {
        "Bug Tracker": "package issues URL",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    packages = setuptools.find_packages(where=".aang"),
    python_requires = ">=3.7"
)
