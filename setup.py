from distutils.core import setup
setup(
    name='nsml visualization',
    install_requires =[
        'visdom',
        'pillow',
        'tensorboardX==1.2',
        'termcolor',
        'psutil',
        'seaborn',
        'cython',
        'scikit-image'
    ]
)
