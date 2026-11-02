"""Derive the 9:16 (1080x1920) Shorts layout from promo.html: same timeline, recomposed scenes."""
s = open('promo.html', encoding='utf-8').read()

CSS = """
<style>
/* ===== vertical 1080x1920 overrides (Shorts safe zone ~ y 150..1440) ===== */
#stage { width: 1080px; height: 1920px; }
#rows { inset: -80px -60px; }
#s1cap { top: 600px; font-size: 54px; }
#s1word { top: 700px; }
#s1label { top: 1030px; }
#s1line { top: 1090px; }
#s2logo { top: 430px; }
#s2ring1, #s2ring2 { top: 580px; }
#s2title { top: 745px; }
#s2sub { top: 950px; font-size: 30px; }
#s2tag1, #s2tag2 { top: 1070px; font-size: 48px; padding: 0 70px; line-height: 1.3; }
#s3words { left: 70px; top: 150px; }
.s3w { font-size: 86px; }
#w2 { top: 100px !important; } #w3 { top: 200px !important; } #w4 { top: 300px !important; }
.s3w small { font-size: 26px; top: 10px; margin-right: 18px; }
#s3foot { left: 74px; top: 560px; font-size: 36px; }
#ui { left: 45px; top: 650px; }
#s4title { top: 150px; font-size: 72px; }
.pcard { width: 900px; padding: 36px 44px; }
#pcL { left: 90px !important; top: 290px; height: 480px; }
#pcR { left: 90px !important; top: 800px; height: 560px; }
.pcard .big { font-size: 110px; margin-top: 22px; }
.pcard .line { font-size: 30px; margin-top: 14px; }
.pcard .chips { left: 44px; right: 44px; bottom: 36px; gap: 10px; }
.pchip { font-size: 20px; padding: 8px 14px; }
#s4foot { top: 1390px; font-size: 38px; padding: 0 60px; }
.feat .txt { left: 80px; top: 230px; width: 920px; }
.feat h2 { font-size: 92px; margin-top: 16px; }
.feat p { font-size: 40px; margin-top: 22px; line-height: 1.3; }
.feat .vis { left: 90px; top: 740px; width: 900px; height: 700px; padding: 40px 44px; }
#cv1 { left: 90px !important; } #cvArrow { left: 405px !important; } #cv2 { left: 564px !important; }
#f3 .keep, #f3 .fmts { left: 78px !important; }
#s6title { top: 160px; font-size: 76px; }
.gchip { width: 460px; height: 104px; }
.gchip .lb { font-size: 28px; }
#s6foot { top: 1330px; font-size: 42px; }
#s7num { top: 620px; font-size: 280px; height: 300px; }
#s7inf { top: 620px; }
#s7lang { top: 950px; font-size: 60px; }
#s7any { top: 1110px; font-size: 50px; padding: 0 80px; line-height: 1.3; }
#s7word { top: 700px; font-size: 190px; }
#s7label { top: 1030px; }
#s8logo { top: 380px; }
#s8title { top: 610px; }
#s8sub { top: 785px; font-size: 28px; }
#s8tag { top: 860px; font-size: 54px; }
#s8pills { top: 1000px; flex-wrap: wrap; padding: 0 200px; row-gap: 14px; }
#s8gh { top: 1200px; }
#s8gh div { font-size: 28px; padding: 16px 24px; }
</style>
</head>"""


def rep(old, new, count=1):
    global s
    assert s.count(old) >= 1, old
    s = s.replace(old, new, count)


rep('</head>', CSS)
rep('const NROWS = 26;', 'const NROWS = 44;')
rep('const UI_S = 1.2;', 'const UI_S = 1.1;')
rep("d.style.left = (135 + (n % 4) * 420) + 'px';", "d.style.left = (70 + (n % 2) * 480) + 'px';")
rep("d.style.top = (270 + Math.floor(n / 4) * 150) + 'px';", "d.style.top = (300 + Math.floor(n / 2) * 124) + 'px';")
rep("const cx = parseFloat(c.style.left) + 195 - 960, cy = parseFloat(c.style.top) + 55 - 540;",
    "const cx = parseFloat(c.style.left) + 230 - 540, cy = parseFloat(c.style.top) + 52 - 960;")
# Long words must fit the narrow frame: wrap in a span, measure once per word, scale down.
rep("window.render = render;", """function setWord(el, txt) {
  if (el._txt !== txt) {
    el._txt = txt;
    el.innerHTML = `<span style="display:inline-block">${txt}</span>`;
    el._fit = Math.min(1, 960 / el.firstChild.offsetWidth);
  }
}
window.render = render;""")
s = s.replace("w.textContent = c.w; l.textContent = c.l;", "setWord(w, c.w); l.textContent = c.l;")
rep("s: grow * lerp(1.04, 1, k)", "s: w._fit * grow * lerp(1.04, 1, k)")
rep("s: 1 + .15 * P(t, 52, 53.75)", "s: w._fit * (1 + .15 * P(t, 52, 53.75))")
open('promo_vertical.html', 'w', encoding='utf-8').write(s)
print('ok')
