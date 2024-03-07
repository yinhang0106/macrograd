from setuptools import setup, find_packages

setup(
    name='macrograd',
    version='0.1',
    packages=find_packages(),
    license='MIT',
    description='A minimalistic autograd engine',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',  
)