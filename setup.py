from setuptools import setup, find_packages

setup(
    name="march-madness-predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.1",
        "beautifulsoup4>=4.9.3",
        "kenpompy>=0.3.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.1",
        "tqdm>=4.61.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="NCAA March Madness Tournament Predictor using Machine Learning",
    keywords="basketball, ncaa, march madness, prediction, machine learning",
    url="https://github.com/your-username/march-madness-predictor",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Sports Analysts",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)