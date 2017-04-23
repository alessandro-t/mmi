from setuptools import setup
from setuptools import find_packages

setup(name='mmi',
      version='1.0',
      description='Multi-multi instance networks in Tensorflow',
      author='Alessandro Tibo',
      author_email='alessandro.tibo@unifi.it',
      url='https://github.com/alessandro-t/mmi',
      download_url='https://github.com/alessandro-t/mmi',
      license='MIT',
      install_requires=['numpy',
                        'tensorflow',
                        'scipy'
                        ],
      package_data={'mmi': ['README.md']},
      packages=find_packages())
