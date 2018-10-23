from distutils.core import setup
import setuptools

setup(
    name='ACE-Defense',
    version='1.0',
    description='Experiment settings for ACE: Artificial Checkerboard Enhancer to Induce and Evade Adversarial Attacks (2018)',
    install_requires=[
        'pillow',
        'torch',
        'torchvision',
        'tensorboardX==1.2',
        'termcolor',
        'scikit-image'
    ]
)
