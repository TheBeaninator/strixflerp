"""tools/flerp-stage.py and [build.path_pins] (card fe4d694019): the symbol-names-only verdict and the bwrap view."""
import importlib.util
import json
import os
import struct
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from flerplib import assemble_zig  # noqa: E402

spec = importlib.util.spec_from_file_location("flerp_stage", os.path.join(ROOT, "tools", "flerp-stage.py"))
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def elf(strtab, text=b"\x90" * 16):
    """A minimal ELF64 LE file: sections .text, .strtab, .shstrtab (enough for elf_sections)."""
    shstr = b"\0.text\0.strtab\0.shstrtab\0"
    body = text + strtab + shstr
    off_text, off_str, off_sh = 64, 64 + len(text), 64 + len(text) + len(strtab)
    shoff = 64 + len(body)
    head = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9)
    head += struct.pack("<HHIQQQIHHHHHH", 2, 62, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 4, 3)
    sh = [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    for name, typ, off, size in ((1, 1, off_text, len(text)), (7, 3, off_str, len(strtab)), (15, 3, off_sh, len(shstr))):
        sh.append(struct.pack("<IIQQQQIIQQ", name, typ, 0, 0, off, size, 0, 0, 1, 0))
    return head + body + b"".join(sh)


class SymbolNamesOnly(unittest.TestCase):
    def pair(self, a, b):
        d = tempfile.mkdtemp()
        pa, pb = os.path.join(d, "a"), os.path.join(d, "b")
        open(pa, "wb").write(a)
        open(pb, "wb").write(b)
        return pa, pb

    def test_thread_id_bits_only(self):
        pa, pb = self.pair(elf(b"\0f\0__anon_335559635\0"), elf(b"\0f\0__anon_805321683\0"))
        self.assertIn("1 __anon_<n> names", stage.symbol_names_only(pa, pb))

    def test_low_bits_differ(self):
        pa, pb = self.pair(elf(b"\0f\0__anon_335559635\0"), elf(b"\0f\0__anon_335559636\0"))
        self.assertIsNone(stage.symbol_names_only(pa, pb))

    def test_code_differs(self):
        pa, pb = self.pair(elf(b"\0__anon_335559635\0"), elf(b"\0__anon_805321683\0", text=b"\xcc" * 16))
        self.assertIsNone(stage.symbol_names_only(pa, pb))

    def test_not_elf(self):
        pa, pb = self.pair(b"abc", b"abd")
        self.assertIsNone(stage.symbol_names_only(pa, pb))


class PathPins(unittest.TestCase):
    def test_errors(self):
        self.assertEqual(assemble_zig.path_pin_errors(None), [])
        self.assertTrue(assemble_zig.path_pin_errors({"tree": "/a"}))
        self.assertTrue(assemble_zig.path_pin_errors({"tree": "rel", "global_cache": "/b"}))
        self.assertTrue(assemble_zig.path_pin_errors({"tree": "/a", "global_cache": "/a/c"}))
        self.assertEqual(assemble_zig.path_pin_errors({"tree": "/a/wt", "global_cache": "/a/.zig-global-wt"}), [])

    def test_view_creates_nothing(self):
        host = tempfile.mkdtemp()
        os.makedirs(os.path.join(host, "plans", "other"))
        out = os.path.join(host, "build")
        os.makedirs(os.path.join(out, "tree"))
        pins = {"tree": os.path.join(host, "plans", "gone", "wt"), "global_cache": os.path.join(host, "plans", "gone", ".zg")}
        json.dump({"path_pins": pins}, open(os.path.join(out, "assembly.json"), "w"))
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(assemble_zig.path_view(out), 0)
        args = buf.getvalue().split("\0")[:-1]
        self.assertEqual(args[args.index("--tmpfs") + 1], os.path.join(host, "plans"))
        self.assertIn(os.path.join(host, "plans", "other"), args)
        self.assertEqual(args[-3:], ["--chdir", pins["tree"], "--"])
        self.assertFalse(os.path.exists(os.path.join(host, "plans", "gone")))


if __name__ == "__main__":
    unittest.main()
