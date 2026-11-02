import os
import sys, pathlib
from playwright.sync_api import sync_playwright
times = [float(x) for x in sys.argv[1:]]
out = pathlib.Path(os.environ.get('OUT','stills')); out.mkdir(exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': int(os.environ.get('W',1920)), 'height': int(os.environ.get('H',1080))})
    msgs = []
    pg.on('console', lambda m: msgs.append(m.text))
    pg.on('pageerror', lambda e: msgs.append('ERR ' + str(e)))
    pg.goto(pathlib.Path(os.environ.get('HTML','promo.html')).resolve().as_uri())
    pg.wait_for_function('window.READY === true', timeout=60000)
    for t in times:
        pg.evaluate(f'render({t})')
        pg.locator('#stage').screenshot(path=str(out / f't{t:05.2f}.png'))
    b.close()
    for m in msgs: print(m)
