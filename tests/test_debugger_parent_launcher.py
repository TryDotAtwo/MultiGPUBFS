import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "host_gate", Path(__file__).parents[1] / "kaggle/host-sized-nccl-gate/kernel.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class DebuggerParentLauncherTests(unittest.TestCase):
    def test_script_target_uses_verified_elf_parent_without_changing_arguments(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "sanitizer"
            parent = Path(root) / "env"
            script.write_bytes(b"#!/bin/sh\nexec sanitizer.real \"$@\"\n")
            parent.write_bytes(b"\x7fELFstub")
            argv = [str(script), "--tool", "memcheck", "app with spaces", "--exact"]
            self.assertEqual(gate.debugger_parent_argv(argv, parent), [str(parent), *argv])

    def test_elf_target_needs_no_parent(self):
        with tempfile.TemporaryDirectory() as root:
            binary = Path(root) / "sanitizer"
            binary.write_bytes(b"\x7fELFstub")
            self.assertEqual(gate.debugger_parent_argv([str(binary), "arg"], None),
                             [str(binary), "arg"])

    def test_non_elf_parent_is_rejected_before_debugger_launch(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "script"
            script.write_bytes(b"#!/bin/sh\n")
            with self.assertRaisesRegex(RuntimeError, "DEBUGGER_PARENT_NOT_ELF"):
                gate.debugger_parent_argv([str(script)], script)


if __name__ == "__main__":
    unittest.main()
