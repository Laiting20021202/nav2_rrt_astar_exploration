"""Setuptools entry point for goal_seeker_rl."""

import glob
import os

from setuptools import find_packages, setup


package_name = "goal_seeker_rl"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob.glob(os.path.join("launch", "*.launch.py"))),
        (f"share/{package_name}/rviz", glob.glob(os.path.join("rviz", "*.rviz"))),
        (f"share/{package_name}/urdf", glob.glob(os.path.join("urdf", "*.urdf"))),
    ],
    install_requires=["setuptools", "numpy", "torch"],
    zip_safe=True,
    maintainer="david",
    maintainer_email="david@localhost",
    description="TD3-based goal navigation package for TurtleBot3 Waffle in ROS 2 Humble.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "goal_seeker_main = goal_seeker_rl.main_node:main",
        ],
    },
)
