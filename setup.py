import setuptools

# with open("README.md", "r", encoding = "utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name = "aang",
    version = "0.0.1",
    author = "seba3y",
    author_email = "sebaeymohamed43@gmail.com",
    description = "speech translation",
  
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    packages = ["torch>=1.10", "torchaudio", "torchtext", "pytorch-lightning>=1.9", "tokenizers",
                "pyarabic"],
    python_requires = ">=3.7"
)
