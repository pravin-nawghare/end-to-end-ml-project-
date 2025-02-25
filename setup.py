from setuptools import find_packages, setup
from typing import List

hypen_e_dot = '-e .'
def get_requirements(file_path:str) ->List[str]:
    """
    This function will return a list of requirements
    """
    requirement = []
    with open(file_path) as file_obj:
        requirement = file_obj.readlines()
        [req.replace("\n","") for req in requirement]

        if hypen_e_dot in requirement:
            requirement.remove(hypen_e_dot)

    return requirement

setup(
    name = "mlproject",
    version = '0.0.1',
    author = "Pravin",
    author_email = "nawgharepravin0@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt'),
)

