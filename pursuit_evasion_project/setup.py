from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
ASSETS = [
    str(path.relative_to(ROOT))
    for path in (ROOT / "assets").glob("*")
    if path.is_file()
]

setup(
    name="pursuit_evasion",
    version="0.3.0",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    data_files=[("pursuit_evasion/assets", ASSETS)],
    install_requires=[
        "gymnasium>=0.29.1",
        "stable-baselines3>=2.2.1",
        "mujoco>=3.1.0",
        "pybullet>=3.2.6",
        "numpy",
        "matplotlib",
        "tensorboard",
        "imageio[ffmpeg]>=2.31",
    ],
)