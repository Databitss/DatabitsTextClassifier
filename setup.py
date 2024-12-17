from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

setup(
    name='databits', 
    version='1.0.5',   
    packages=find_packages(),  
    install_requires=read_requirements('requirements.txt'),  
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown', 
    author='Databits Team',
    author_email='databitsteam@gmail.com',
    description='Text Classifier using LSTM, GRU, and Transformer BERT',
    url='https://github.com/Databitss/databits/',  
    classifiers=[  
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
    ],
)
