"""Check built HTML for missing local targets, anchors and image alt text."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import json
import yaml
R=Path(__file__).resolve().parents[1]
config=yaml.safe_load((R/'mkdocs.yml').read_text())
S=R/config.get('site_dir','site')
base_path=unquote(urlsplit(config.get('site_url') or '').path).rstrip('/')
class Page(HTMLParser):
 def __init__(self):super().__init__();self.ids=set();self.refs=[];self.noalt=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.add(a['id'])
  if tag=='a' and 'href' in a:self.refs.append(a['href'])
  if tag in ['img','script'] and 'src' in a:self.refs.append(a['src'])
  if tag=='link' and 'href' in a:self.refs.append(a['href'])
  if tag=='img' and 'alt' not in a:self.noalt.append(a.get('src'))
pages={}
for p in S.rglob('*.html'):
 h=Page();h.feed(p.read_text());pages[p.resolve()]=h
if not pages:raise SystemExit('No built HTML found. Run python -m mkdocs build --strict first.')
errors=[];refs=0
for p,h in pages.items():
 for ref in h.refs:
  u=urlsplit(ref)
  if u.scheme or u.netloc:continue
  refs+=1
  ref_path=unquote(u.path)
  if ref_path.startswith('/'):
   if base_path and ref_path!=base_path and not ref_path.startswith(base_path+'/'):
    errors.append(f'{p.relative_to(S.resolve())}: outside site URL path {ref}')
    continue
   path=(S/ref_path[len(base_path):].lstrip('/')).resolve()
  else:
   path=(p.parent/ref_path).resolve() if ref_path else p
  if path.is_dir():path/= 'index.html'
  if not path.exists():errors.append(f'{p.relative_to(S.resolve())}: missing {ref}')
  elif u.fragment and path in pages and unquote(u.fragment) not in pages[path].ids:
   errors.append(f'{p.relative_to(S.resolve())}: missing anchor {ref}')
 for img in h.noalt:errors.append(f'{p.name}: missing alt {img}')
report={'html_pages':len(pages),'local_references_checked':refs,'errors':errors}
print(json.dumps(report,indent=2))
if errors:raise SystemExit(1)
