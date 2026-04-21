from setuptools import find_packages, setup

package_name = "mapless_nav2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/launch",
            [
                "launch/mapless_tb3_sim.launch.py",
                "launch/frontier_explore_tb3.launch.py",
            ],
        ),
        (
            f"share/{package_name}/config",
            [
                "config/nav2_mapless_params.yaml",
                "config/nav2_frontier_exploration_params.yaml",
                "config/exploration_manager.yaml",
                "config/slam_toolbox_online_async.yaml",
            ],
        ),
        (
            f"share/{package_name}/rviz",
            [
                "rviz/mapless_nav2.rviz",
                "rviz/frontier_explore.rviz",
            ],
        ),
        (
            f"share/{package_name}/behavior_trees",
            ["behavior_trees/explore_goal_switch_nav.xml"],
        ),
        (
            f"share/{package_name}/models/turtlebot3_waffle_45deg",
            [
                "models/turtlebot3_waffle_45deg/model.sdf",
                "models/turtlebot3_waffle_45deg/model-1_4.sdf",
                "models/turtlebot3_waffle_45deg/model.config",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="david",
    maintainer_email="david@example.com",
    description="Mapless navigation workflow built on Nav2.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mapless_goal_manager = mapless_nav2.mapless_goal_manager:main",
            "mapless_send_goal = mapless_nav2.send_goal:main",
            "mapless_safety_controller = mapless_nav2.safety_controller:main",
            "mapless_scan_stabilizer = mapless_nav2.scan_stabilizer:main",
            "exploration_coordinator = mapless_nav2.exploration_coordinator:main",
        ],
    },
)
