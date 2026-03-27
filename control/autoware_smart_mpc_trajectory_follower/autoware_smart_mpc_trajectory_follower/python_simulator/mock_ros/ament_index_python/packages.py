"""Mock get_package_share_directory for running smart-MPC without ROS2."""
import os

# The package share directory is the Python package root that contains param/.
_PACKAGE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../autoware_smart_mpc_trajectory_follower")
)

_SHARE_MAP = {
    "autoware_smart_mpc_trajectory_follower": _PACKAGE_ROOT,
}


def get_package_share_directory(package_name: str) -> str:
    if package_name in _SHARE_MAP:
        return _SHARE_MAP[package_name]
    raise KeyError(f"Package '{package_name}' not found in mock ament index")
