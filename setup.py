from setuptools import setup

def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires

INSTALL_REQUIRES = parse_requirements_file("requirements.txt")
setup(
    name="object_removal",    
    install_requires=INSTALL_REQUIRES
)