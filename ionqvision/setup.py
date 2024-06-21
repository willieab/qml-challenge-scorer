#
# IonQ, Inc., Copyright (c) 2024,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at Quantum Korea hosted by SKKU at Hotel Samjung and only during the
# June 21-23, 2024 duration of such event.
#
from setuptools import find_packages, setup


def get_requirements():
    with open("requirements.txt", "r") as req:
        ret = [line.strip() for line in req]
    return ret


with open("README", "r") as f:
    longdescription = f.read()


setup(
    name="ionqvision",
    url="https://github.com/ionq-applications/ionq-vision-challenge",
    version="1.0",
    author="Willie Aboumrad, IonQ Inc.",
    description="IonQ's module for hybrid quantum-classical neural networks",
    longdescription=longdescription,
    author_email="aboumrad@ionq.co",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=get_requirements(),
)
