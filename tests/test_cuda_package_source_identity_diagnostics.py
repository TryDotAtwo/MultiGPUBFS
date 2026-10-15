import importlib.util, pathlib, subprocess, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("package_variants",ROOT/"scripts/package_diagnostic_variants.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class SourceIdentity(unittest.TestCase):
 def test_build_configuration_is_separate_but_kernel_changes_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);subprocess.run(["git","init","-q",d],check=True);subprocess.run(["git","-C",d,"config","user.name","fixture"],check=True);subprocess.run(["git","-C",d,"config","user.email","fixture@example.invalid"],check=True)
   cuda=root/"cuda";cuda.mkdir();(cuda/"kernel.cu").write_text("kernel version1");(cuda/"CMakeLists.txt").write_text("static default")
   def commit():
    subprocess.run(["git","-C",d,"add","."],check=True);subprocess.run(["git","-C",d,"commit","-qm","fixture"],check=True);return subprocess.check_output(["git","-C",d,"rev-parse","HEAD"],text=True).strip()
   first=commit();(cuda/"CMakeLists.txt").write_text("shared default");second=commit()
   a=module.cuda_source_identity(root,first);b=module.cuda_source_identity(root,second)
   self.assertNotEqual(a["cuda_source_tree"],b["cuda_source_tree"]);self.assertEqual(a["cuda_kernel_source_sha256"],b["cuda_kernel_source_sha256"])
   (cuda/"kernel.cu").write_text("kernel version2");third=commit();self.assertNotEqual(b["cuda_kernel_source_sha256"],module.cuda_source_identity(root,third)["cuda_kernel_source_sha256"])
   (cuda/"new-helper.h").write_text("helper");fourth=commit();self.assertNotEqual(module.cuda_source_identity(root,third)["cuda_kernel_source_sha256"],module.cuda_source_identity(root,fourth)["cuda_kernel_source_sha256"])
