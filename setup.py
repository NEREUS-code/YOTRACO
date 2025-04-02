from setuptools import setup, find_packages

setup(
    name="yotraco",
    version="0.2.4",  # Updated version
    packages=find_packages(include=["YOTRACO*"], exclude=["images"]),  # Match the inclusion/exclusion from pyproject
    install_requires=[
        "torch",
        "opencv-python",
        "ultralytics",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    author="YOTRACO TEAM",
    author_email="nereuscode@gmail.com",
    description="A YOLO-based object tracking and counting system with customizable tracking lines and movement direction.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=['python', 'video', 'yolo', 'openCV'],
    url="https://github.com/NEREUS-code/YOTRACO",
    python_requires=">=3.8",
)
