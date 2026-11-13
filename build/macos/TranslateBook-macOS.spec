# -*- mode: python ; coding: utf-8 -*-
import os
import tempfile
import tiktoken
import tiktoken_ext.openai_public

block_cipher = None

# tiktoken_ext only holds the encoding registry (Python code). The encoding
# file itself is downloaded on first use, which fails offline or behind a
# firewall (#339). Download it now, at build time, into a dedicated cache and
# bundle that cache; launcher.py points TIKTOKEN_CACHE_DIR at it at runtime.
tiktoken_ext_dir = os.path.dirname(tiktoken_ext.openai_public.__file__)
tiktoken_cache_dir = tempfile.mkdtemp(prefix='tbl_tiktoken_cache_')
os.environ['TIKTOKEN_CACHE_DIR'] = tiktoken_cache_dir
tiktoken.get_encoding('cl100k_base')
if not os.listdir(tiktoken_cache_dir):
    raise SystemExit('tiktoken did not write the cl100k_base encoding to the build cache')

# Prepare datas list
datas_list = [
    ('../../src/web/static', 'src/web/static'),
    ('../../src/web/templates', 'src/web/templates'),
    ('../../src', 'src'),
    ('../../.env.example', '.'),
]

# Bundle the style presets only when some are shipped. PyInstaller treats a
# missing data source as a hard error, and the repo currently ships none, so an
# unconditional entry breaks the build. launcher.py already skips the first-run
# seeding when the bundle carries no Custom_Instructions folder.
custom_instructions_dir = os.path.join(SPECPATH, '..', '..', 'Custom_Instructions')
if os.path.isdir(custom_instructions_dir):
    datas_list.append((custom_instructions_dir, 'Custom_Instructions'))

# Add tiktoken_ext if directory exists
if os.path.exists(tiktoken_ext_dir):
    datas_list.append((tiktoken_ext_dir, 'tiktoken_ext/openai_public'))

# Bundled encoding cache (see the top of this file)
datas_list.append((tiktoken_cache_dir, 'tiktoken_cache'))

a = Analysis(
    ['../../launcher.py'],
    pathex=[],
    binaries=[],
    datas=datas_list,
    hiddenimports=[
        'flask',
        'flask_cors',
        'flask_socketio',
        'python_socketio',
        'socketio',
        'engineio',
        'engineio.async_drivers.threading',
        'requests',
        'tqdm',
        'httpx',
        'lxml',
        'lxml.etree',
        'lxml._elementpath',
        'dotenv',
        'aiofiles',
        'tiktoken',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        'pyyaml',
        'jinja2',
        'langdetect',
        'PIL',
        'pymupdf',
        'dns',
        'dns.resolver',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TranslateBook',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=True,  # Enable for macOS compatibility
    target_arch=None,  # Will build for current architecture (arm64 or x86_64)
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
