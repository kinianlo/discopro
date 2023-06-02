from setuptools import setup, find_packages

setup(
    name='discopro',
    version='0.1.4',
    url='https://github.com/kinianlo/discopy-pronoun',
    author='Kin Ian Lo',
    author_email='keonlo123@gmail.com',
    description='A plugin package for discopy to handle anaphora',
    packages=find_packages(),    
    install_requires=['discopy'],
)
