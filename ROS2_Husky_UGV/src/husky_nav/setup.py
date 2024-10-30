from setuptools import setup, find_packages

package_name = 'husky_nav'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),  # This will find the package automatically
    data_files=[
        ('share/' + package_name + '/launch', ['launch/husky_gazebo.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='A package to navigate the Husky UGV in a custom Gazebo environment.',
    license='License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Add any executable scripts here if applicable
        ],
    },
)
