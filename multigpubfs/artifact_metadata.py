"""Read requirements of already-built ELF artifacts; packaging only."""
import re,subprocess
def elf_runtime_requirements(paths):
 versions={name:[] for name in ('GLIBC','GLIBCXX','CXXABI')}
 for path in paths:
  result=subprocess.run(['readelf','--version-info',str(path)],capture_output=True,text=True,check=True,timeout=30)
  for name,values in versions.items():values.extend(re.findall(r'\b'+name+r'_([0-9]+(?:\.[0-9]+)+)\b',result.stdout))
 return {name:max(values,key=lambda text:tuple(map(int,text.split('.')))) if values else None for name,values in versions.items()}
