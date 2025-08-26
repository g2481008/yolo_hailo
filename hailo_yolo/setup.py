from setuptools import find_packages, setup

package_name = 'hailo_yolo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/hailo_yolo.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='g2012013acsl@gmail.com',
    description='YOLO inference with Hailo-8 on RealSense image via ROS2',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo = hailo_yolo.yolo:main',
            'yolo_overlay_node = hailo_yolo.yolo_overlay:main'
        ],
    },
)
