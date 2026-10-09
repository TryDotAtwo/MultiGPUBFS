from pathlib import Path
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel
class NativeWheel(bdist_wheel):
 def finalize_options(self):
  super().finalize_options()
  if (Path(__file__).parent/'multigpubfs/_native/bin/mgbfs').is_file():self.root_is_pure=False
 def get_tag(self):
  python,abi,platform=super().get_tag()
  return ('py3','none',platform) if not self.root_is_pure else (python,abi,platform)
setup(cmdclass={'bdist_wheel':NativeWheel})
