#nsml: nsml/default_ml:cuda9_torch1.0
from distutils.core import setup
setup(
    name='nsml visualization',
    install_requires = [
        'visdom',
        'pillow',
        'tensorboardX==1.2',
        'termcolor',
        'scikit-image',
    ]
)
