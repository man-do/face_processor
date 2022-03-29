from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['face_processor'],
    #scripts=['scripts/face_mesh.py', 'scripts/face_processor_node.py'],
    package_dir={'': 'src'}
)

setup(**d)