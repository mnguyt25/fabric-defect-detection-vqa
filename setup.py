"""Script cài đặt dự án"""

from setuptools import setup, find_packages

setup(
    name="scratch-inspection-system",
    version="1.0.0",
    author="Your Name",
    description="Hệ thống phát hiện và phân vùng vết trầy xước trên vật liệu nhựa",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "scratch-inspect=main:main",
        ],
    },
)