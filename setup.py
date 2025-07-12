import setuptools

setuptools.setup(
    name="MNIST_AI_Training",
    version="0.1.0",
    author="Brandon Young",
    description="An exercise building AI models such as KNN, RF, NN, and CNN",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "numba",
        "scikit-learn",
        "matplotlib",
        "cupy_cuda12x",
        "pandas",
        "setuptools",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
