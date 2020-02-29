from setuptools import find_packages, setup

setup(
    name='grog',
    packages=['grog', 'grog.models', 'grog.cmd', 'grog.evaluation'],
    version='0.1.0',
    description='Bachelor thesis for speech separation (using deep-clustering)',
    author='maxammann',
    license='MIT'
)
