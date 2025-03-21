# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[
        ('C_engine/src/libanalysis_engine.dll', '.'),
        ('C:/mingw64/bin/libgcc_s_seh-1.dll', '.'),
        ('C:/mingw64/bin/libwinpthread-1.dll', '.'),
        ('C:/mingw64/bin/libgomp-1.dll', '.')
    ],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('analysis.py', '.'),
        ('database.py', '.'),
        ('config.py', '.')
    ],
    hiddenimports=['flask_session'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TotoAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None
)
