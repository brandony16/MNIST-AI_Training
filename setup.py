import setuptools

setuptools.setup(
    name="MNIST_AI_Training",
    version="0.1.0",
    author="Your Name",
    description="A fast Random Forest implementation with Numba & NumPy",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=["numpy", "numba", "joblib", "scikit-learn"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
