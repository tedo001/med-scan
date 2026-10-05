# PyInstaller spec for the slim MEDSCAN bundle (built-in engine; see requirements-app.txt).
#
#     pip install -r requirements-app.txt pyinstaller
#     pyinstaller packaging/medscan.spec
#
# dist/MEDSCAN/MEDSCAN(.exe) then runs without Python. Check a build with:
#     set MEDSCAN_SMOKE=1 && dist\MEDSCAN\MEDSCAN.exe
# -*- mode: python ; coding: utf-8 -*-
import os

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

a = Analysis(
    [os.path.join(ROOT, "medscan.py")],
    pathex=[ROOT],
    datas=[
        (os.path.join(ROOT, "ui", "assets"), os.path.join("ui", "assets")),
        (os.path.join(ROOT, "msx", "knowledge"), os.path.join("msx", "knowledge")),
        (os.path.join(ROOT, "samples"), "samples"),
    ],
    hiddenimports=["pydicom.encoders", "scipy.ndimage"],
    excludes=["torch", "torchvision", "torchxrayvision", "matplotlib", "tkinter"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="MEDSCAN", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="MEDSCAN")
