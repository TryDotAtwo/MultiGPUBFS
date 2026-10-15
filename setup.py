from pathlib import Path
import json,subprocess
from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel
class PackageBuild(build_py):
 def run(self):
  super().run()
  try:source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True,stderr=subprocess.DEVNULL,timeout=5).strip()
  except (OSError,subprocess.SubprocessError):source=None
  target=Path(self.build_lib)/'multigpubfs/_package_info.json';target.parent.mkdir(parents=True,exist_ok=True)
  target.write_text(json.dumps({'schema':1,'source_commit':source,'scope':'Python package build identity; native component source identities are separate'}))
class NativeWheel(bdist_wheel):
 def finalize_options(self):
  super().finalize_options()
  if (Path(__file__).parent/'multigpubfs/_native/bin/mgbfs').is_file():self.root_is_pure=False
 def get_tag(self):
  python,abi,platform=super().get_tag()
  return ('py3','none',platform) if not self.root_is_pure else (python,abi,platform)
setup(cmdclass={'bdist_wheel':NativeWheel,'build_py':PackageBuild})
