"""Editable documentation schematics; no simulation or scientific estimator."""
from pathlib import Path
from html import escape
R=Path(__file__).resolve().parents[1]/'docs/assets'
def svg(name,w,h,body,title):
 R.joinpath(name).write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img"><title>{escape(title)}</title><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#56766d"/></marker></defs><style>text{{font-family:system-ui,sans-serif;fill:#183c42;font-size:17px}}.small{{font-size:14px;fill:#4a6467}}.bold{{font-weight:700}}.line{{stroke:#56766d;stroke-width:2;fill:none;marker-end:url(#arrow)}}</style><rect width="100%" height="100%" fill="white"/>{body}</svg>''')
def text(x,y,s,cls='',anchor='start'):
 return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>'
def box(x,y,w,h,lines,fill='#edf4ef'):
 s=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="#bccfc5"/>'
 for i,line in enumerate(lines):s+=text(x+w/2,y+27+24*i,line,'bold' if i==0 else 'small','middle')
 return s
b=text(28,32,'Detector geometry','bold')+text(28,56,'Side view · schematic, not to scale','small')
b+='<circle cx="80" cy="140" r="25" fill="#f3ddb0" stroke="#c7a76b"/>'
b+=text(80,192,'Source','bold','middle')+text(80,214,'centre z = −200 mm','small','middle')
b+='<path class="line" d="M110 140 L310 140"/>'
b+=text(203,115,'incident gamma','small','middle')
for x,w,fill in [(325,75,'#cce0e7'),(400,230,'#d7e5d2'),(630,12,'#cce0e7'),(642,24,'#eac47c')]:
 b+=f'<rect x="{x}" y="80" width="{w}" height="125" fill="{fill}" stroke="#527a72"/>'
b+=text(362,240,'Front glass','bold','middle')+text(362,262,'4 mm','small','middle')
b+=text(515,122,'Crystals','bold','middle')+text(515,148,'15 mm','', 'middle')
b+='<path class="line" d="M404 179 L617 179"/>'+text(515,173,'increasing z','small','middle')
b+='<path d="M636 81 L698 62 L755 62" fill="none" stroke="#527a72"/>'
b+=text(760,67,'Rear glass · 0.1 mm','small')
b+='<path d="M654 158 L717 190 L755 190" fill="none" stroke="#527a72"/>'
b+=text(760,194,'SiPM · 0.5 mm','small')
b+=text(400,224,'−7.5','small','middle')+text(630,224,'+7.5','small','middle')
b+=text(514,245,'crystal-face z (mm)','small','middle')
b+='<path d="M80 282 L80 295 L515 295 L515 282" fill="none" stroke="#7a9690"/>'
b+=text(297,318,'200 mm · source centre to crystal centre','small','middle')
b+=text(28,365,'Plan view · looking along z','bold')
for iy in range(8):
 for ix in range(8):
  b+=f'<rect x="{30+ix*18}" y="{385+iy*18}" width="15" height="15" fill="#d7e5d2" stroke="#789781"/>'
b+=text(207,410,'8 × 8 crystals, each 3 × 3 mm across','bold')
b+=text(207,443,'Pitch: 3.2 mm  |  Gap: 0.2 mm','small')
b+=text(207,476,'Crystal-array outer span: 25.4 mm','small')
b+=text(207,509,'Glass and readout width: 25.8 mm','small')
svg('geometry.svg',970,555,b,'Current module geometry and coordinate convention')
b=text(28,32,'What is measured, and what is truth?','bold')
b+=box(310,56,290,58,['Simulated gamma interaction'])
b+='<path class="line" d="M390 114 L210 150"/><path class="line" d="M520 114 L700 150"/>'
b+=box(40,155,340,80,['Optical branch','Scintillation → entering photons'])
b+=box(530,155,340,80,['Truth branch','Primary gamma step records'])
b+='<path class="line" d="M210 235 L210 272"/><path class="line" d="M700 235 L700 272"/>'
b+=box(40,276,340,82,['Channel assignment + PDE','64 photoelectron counts'])
b+=box(530,276,340,82,['Local-deposit filter + earliest step','Selected x, y, z label'])
b+='<path class="line" d="M210 358 L355 400"/><path class="line" d="M700 358 L560 400"/>'
b+=box(310,403,290,82,['Join within the same job','Feature / truth CSV'])
b+=text(455,521,'LOCAL ROUTE STARTS WITH EXISTING CSV PRODUCTS','bold','middle')
b+=text(455,552,'Separate saved predictions → evaluation of residuals and uncertainty','small','middle')
svg('measurement.svg',910,575,b,'Measurement and truth branches joined into feature CSVs')
b=text(28,32,'The current conditional-density network','bold')
b+=box(306,58,288,70,['8 × 8 photoelectron map','1 input channel'])
for x,k in [(28,2),(325,4),(622,8)]:
 b+=f'<path class="line" d="M450 128 L{x+130} 173"/>'
 b+=box(x,179,260,92,[f'{k} × {k} convolution','8 channels · ReLU','Adaptive pooling → 2 × 2'])
 b+=f'<path class="line" d="M{x+130} 271 L450 312"/>'
b+=box(277,317,346,58,['Concatenate → 96 entries'])
b+='<path class="line" d="M450 375 L450 409"/>'
b+=box(277,413,346,58,['Linear → 64 · ReLU'])
b+='<path class="line" d="M450 471 L450 505"/>'
b+=box(216,509,468,82,['Linear → 4 + 3K outputs','x/y Gaussian parameters + z mixture parameters'],'#faf0d9')
b+=text(450,625,'Read and evaluate saved parameters now; train a model later.','small','middle')
svg('network.svg',910,650,b,'Parallel CNN branches and conditional-density outputs')
print('Wrote three editable SVG diagrams')
