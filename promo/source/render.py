import os
import sys, time, pathlib, subprocess
from playwright.sync_api import sync_playwright
FPS = 30
DUR = 60
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end = int(sys.argv[2]) if len(sys.argv) > 2 else FPS * DUR
out = sys.argv[3] if len(sys.argv) > 3 else 'video_raw.mp4'
ff = subprocess.Popen(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', str(FPS),
                       '-c:v', 'png', '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '12',
                       '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': int(os.environ.get('W',1920)), 'height': int(os.environ.get('H',1080))})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(pathlib.Path(os.environ.get('HTML','promo.html')).resolve().as_uri())
    pg.wait_for_function('window.READY === true', timeout=90000)
    t0 = time.time()
    for i in range(start, end):
        pg.evaluate(f'render({i / FPS})')
        ff.stdin.write(pg.screenshot(type='png', clip={'x': 0, 'y': 0, 'width': int(os.environ.get('W',1920)), 'height': int(os.environ.get('H',1080))}))
        if i % 150 == 0:
            print(f'frame {i}  {time.time() - t0:.0f}s', flush=True)
    b.close()
    if errs: print('ERRORS', errs[:5])
ff.stdin.close(); ff.wait()
print('done', out)
