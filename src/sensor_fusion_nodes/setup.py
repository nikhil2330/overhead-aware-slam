from setuptools import find_packages, setup
from glob import glob


package_name = 'sensor_fusion_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', glob('models/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nikhil',
    maintainer_email='nikhilks29@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'odom_tf = sensor_fusion_nodes.odom_tf:main',
        'object_detection = sensor_fusion_nodes.object_detection:main',
        'object_detction2 = sensor_fusion_nodes.object_detction2:main',
        'rgbd_cam_view = sensor_fusion_nodes.rgbd_cam_view:main',
        'scan_fuser = sensor_fusion_nodes.scan_fuser:main',
        'object_location = sensor_fusion_nodes.object_location:main',
        'occupancy_map = sensor_fusion_nodes.occupancy_map:main',
        'teleop_snapshot = sensor_fusion_nodes.teleop_snapshot:main',
        'capture_session = sensor_fusion_nodes.capture_session:main',
        'yolo_detection = sensor_fusion_nodes.yolo_detection:main',
        'gz_lidar3d_to_pointcloud = sensor_fusion_nodes.gz_lidar3d_to_pointcloud:main',
        'pointcloud_occupancy = sensor_fusion_nodes.pointcloud_occupancy:main',
        'map_filler = sensor_fusion_nodes.map_filler:main',
        'save_map = sensor_fusion_nodes.save_map:main',
        'nav_evaluator = sensor_fusion_nodes.nav_evaluator:main',
    ],
    }
)
