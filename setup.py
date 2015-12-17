from setuptools import setup, Command
import os

class CleanCommand(Command):
    """
    Custom clean command to tidy up the project root.
    """
    
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

#def readme():
    #with open('README.rst') as f:
        #return f.read()

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(name='CRRLpy',
      version='0.1.0b',
      description='Carbon Radio Recombination Line analysis',
      #url='http://github.com/storborg/funniest',
      author='Pedro Salas',
      author_email='psalas@strw.leidenuniv.nl',
      license='MIT',
      packages=['crrlpy', 'crrlpy/models'],
      zip_safe=False,
      install_requires=reqs,
      cmdclass={'clean':CleanCommand})
