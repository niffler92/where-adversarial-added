#nsml: nsml/default_ml:cuda9_torch1.0
from setuptools import setup
setup(
    name='nsml visualization',
    install_requires=[
        'scipy',
        'pillow',
        'termcolor',
        'torch>=1.0.0',
        'torchvision'
    ]
)
