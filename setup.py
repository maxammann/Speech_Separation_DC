from setuptools import find_packages, setup

setup(
    name='grog',
    packages=['grog', 'grog.models', 'grog.cmd', 'grog.evaluation'],
    version='0.1.0',
    description='Bachelor thesis for speech separation (using deep-clustering)',
    author='maxammann',
    license='MIT',
    install_requires=[
        'librosa==0.6.3',
        'scipy==1.4.1',
        'matplotlib',
        'museval',
        'scikit-learn==0.20.3',
        'numpy==1.16.3',
        'tensorflow==1.13.1'
    ]
)
