from setuptools import setup

setup(
    name='keras-multi-head',
    version='0.5',
    packages=['keras_multi_head'],
    url='https://github.com/CyberZHG/keras-multi-head',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='A wrapper layer for stacking layers horizontally',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'Keras',
    ],
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
