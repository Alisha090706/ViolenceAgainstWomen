from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path):
    '''
    This function returns the list of requirements'''
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        return requirements

setup(
    name="Violence Against Women",
    version="0.0.1",
    author="Alisha Kapoor & Bhawya Kumari",
    author_email="alisha021btcseai24@igdtuw.ac.in, bhawya048btcseai24@igdtuw.ac.in",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)