from setuptools import setup

setup(name="tmw",
    version="0.2.1",
    description="Topic Modeling Workflow",
    url="http://github.com/cligs/tmw",
    author="Christof Schöch",
    author_email="c.schoech@gmail.com",
    license="MIT",
    packages=["tmw"],
    install_requires=[
    "pandas>==0.17.0",
    "numpy>==1.9.0A",
    ]
    zip_safe=False)
