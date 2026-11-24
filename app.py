"""Single-user local studio. Bind to loopback; public hosting requires authentication."""
import os, json, sqlite3, shutil, uuid, hashlib, threading, time, re, base64, binascii, secrets
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager
from urllib.parse import urlparse, parse_qs
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import engine
ROOT=Path(os.environ.get('ZENVYRA_DATA_DIR',str(Path(__file__).parent/'data'))).resolve();ROOT.mkdir(parents=True,exist_ok=True)
DB=ROOT/'studio.sqlite';LIMIT=int(os.getenv('ZENVYRA_MAX_UPLOAD_BYTES',str(500_000_000_000)));CHUNK=8*1024*1024
lock=threading.RLock();wake=threading.Event();stop=threading.Event()
@contextmanager
def db():
    c=sqlite3.connect(DB,timeout=30);c.row_factory=sqlite3.Row
    try:yield c;c.commit()
    finally:c.close()
def init():
    with db() as c:
        c.executescript('''PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY, workspace TEXT NOT NULL, name TEXT NOT NULL, url TEXT, size INTEGER DEFAULT 0, fingerprint TEXT, received INTEGER DEFAULT 0, meta TEXT, analysis TEXT, settings TEXT DEFAULT '{}', clips TEXT DEFAULT '[]', created REAL);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, project TEXT NOT NULL, kind TEXT, payload TEXT, state TEXT, progress REAL, message TEXT, result TEXT, created REAL, updated REAL);''')
        c.execute("UPDATE jobs SET state='queued',progress=0,message='Recovered interrupted job' WHERE state='running'")
def row(id):
    with db() as c:r=c.execute('SELECT * FROM projects WHERE id=?',(id,)).fetchone()
    if not r:raise HTTPException(404,'Project not found')
    return dict(r)
def decoded(r):
    r=dict(r)
    for k in ('meta','analysis','settings','clips','payload','result'):
        if k in r and r[k] is not None:r[k]=json.loads(r[k])
    if 'analysis' in r:
        a=r['analysis'];r['analysis']=None if not a else {k:v for k,v in a.items() if k!='samples'}
    return r
def directory(id):return ROOT/id
def src(id):return directory(id)/'source.bin'
def patch(id,**kwargs):
    with db() as c:c.execute('UPDATE projects SET '+','.join(k+'=?' for k in kwargs)+' WHERE id=?',(*kwargs.values(),id))
def job_update(id,**kwargs):
    kwargs['updated']=time.time()
    with db() as c:c.execute('UPDATE jobs SET '+','.join(k+'=?' for k in kwargs)+' WHERE id=?',(*kwargs.values(),id))
def enqueue(id,kind,payload):
    with lock,db() as c:
        encoded=json.dumps(payload,sort_keys=True,separators=(',',':'));old=c.execute("SELECT * FROM jobs WHERE project=? AND kind=? AND payload=? AND state IN ('queued','running','done') ORDER BY created DESC LIMIT 1",(id,kind,encoded)).fetchone()
        if old and (kind!='analyze' or old['state']!='done'):
            if old['state']!='done' or (old['result'] and (directory(id)/json.loads(old['result']).get('file','missing')).is_file()):return decoded(old)
        jid=uuid.uuid4().hex;now=time.time();c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?)',(jid,id,kind,encoded,'queued',0,'Queued',None,now,now))
    wake.set();return {'id':jid,'state':'queued'}
def validate_settings(s):
    count=s.get('count','Auto')
    if count!='Auto' and not 1<=int(count)<=100:raise ValueError('Clip count must be 1–100')
    if not 1<=float(s.get('min_seconds',s.get('seconds',30)))<=float(s.get('seconds',30))<=3600:raise ValueError('Duration must be 1–3600 seconds')
    ratios=s.get('ratios',['9:16']);
    if not isinstance(ratios,list) or not ratios or len(ratios)>9 or len(set(ratios))!=len(ratios) or any(r not in {*engine.RATIOS,'Original','Custom'} for r in ratios):raise ValueError('Invalid aspect ratios')
    if s.get('quality','Source Max') not in ('Source Max','720p','1080p','4K') or s.get('mode','fit') not in ('fit','fill','blur') or s.get('profile','auto') not in ('auto','speaker','gameplay'):raise ValueError('Invalid output settings')
    if 'Custom' in ratios:
        w,h=float(s.get('ratio_w',0)),float(s.get('ratio_h',0))
        if not 0<w<=10000 or not 0<h<=10000 or not .2<=w/h<=5:raise ValueError('Custom ratio must be 1:5 to 5:1')
    return s
def valid(s):
    try:return validate_settings(s)
    except (ValueError,TypeError,OverflowError,ZeroDivisionError) as e:raise HTTPException(422,str(e))
def worker():
    while not stop.is_set():
        with db() as c:
            j=c.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if j:c.execute("UPDATE jobs SET state='running',message='Starting',updated=? WHERE id=?",(time.time(),j['id']))
        if not j:wake.wait(1);wake.clear();continue
        j=dict(j);jid,pid=j['id'],j['project'];payload=json.loads(j['payload']);start=time.monotonic();last=[0]
        def progress(p,m):
            if time.monotonic()-last[0]>.5:job_update(jid,progress=p,message=m);last[0]=time.monotonic()
        try:
            r=row(pid);meta=json.loads(r['meta']);a=json.loads(r['analysis']) if r['analysis'] else None
            if j['kind']=='analyze':
                if not a:a=engine.analyze(src(pid),meta,progress);patch(pid,analysis=json.dumps(a))
                clips=engine.highlights(a,meta,payload);patch(pid,clips=json.dumps(clips));result={'clips':clips,'asr':a['asr'],'reused_analysis':bool(r['analysis'])}
            else:
                settings=payload['settings'];clip=payload['clip'];ratio=payload['ratio'];name=hashlib.sha256(j['payload'].encode()).hexdigest()[:24]+'.mp4';out=directory(pid)/name
                if out.exists():properties=engine.probe(out)
                else:properties=engine.export(src(pid),out,meta,a or {'samples':[],'border':None},clip,ratio,settings,progress)
                result={'file':name,'properties':properties,'ratio':ratio,'clip':clip}
            result['elapsed_seconds']=round(time.monotonic()-start,3);job_update(jid,state='done',progress=1,message='Complete',result=json.dumps(result))
        except Exception as e:job_update(jid,state='failed',message=str(e)[:1800])
@asynccontextmanager
async def lifespan(app):
    init();stop.clear();t=threading.Thread(target=worker,daemon=True);t.start();yield;stop.set();wake.set();t.join(timeout=2)
app=FastAPI(title='Zenvyra Studio',lifespan=lifespan)
# Single-owner hosting: every studio route, including video bytes, is protected.
@app.get('/healthz', include_in_schema=False)
def healthz():return {'status':'ok'}
@app.middleware('http')
async def protect(request,call_next):
    if request.url.path == '/healthz' and request.method in ('GET','HEAD'):
        return await call_next(request)
    hosted=os.getenv('ZENVYRA_HOSTED','').lower() in ('1','true','yes')
    user=os.getenv('ZENVYRA_USERNAME','studio')
    password=os.getenv('ZENVYRA_PASSWORD','')
    if hosted and len(password)<16:
        return JSONResponse({'detail':'Hosting access is not configured'},503)
    if password:
        supplied=request.headers.get('authorization','')
        try:
            scheme, token=supplied.split(' ',1)
            if scheme.lower()!='basic':raise ValueError()
            raw=base64.b64decode(token,validate=True)
            u,p=raw.split(b':',1)
            ok_user=secrets.compare_digest(u,user.encode())
            ok_pass=secrets.compare_digest(p,password.encode())
            authorized=ok_user and ok_pass
        except (ValueError,binascii.Error):authorized=False
        if not authorized:
            return JSONResponse({'detail':'Owner sign-in required'},401,headers={'WWW-Authenticate':'Basic realm="Zenvyra Studio", charset="UTF-8"','Cache-Control':'no-store'})
    if request.method not in ('GET','HEAD','OPTIONS'):
        origin=request.headers.get('origin')
        if origin and urlparse(origin).netloc!=request.headers.get('host'):return __import__('fastapi').responses.JSONResponse({'detail':'Cross-origin write rejected'},403)
    response=await call_next(request)
    if password or hosted:response.headers['Cache-Control']='private, no-store'
    return response
class Project(BaseModel):
    workspace:str;name:str=Field(min_length=1,max_length=150);url:str|None=None
class Begin(BaseModel):
    size:int=Field(gt=0);fingerprint:str=Field(min_length=16,max_length=256)
def youtube_id(url):
    try:
        p=urlparse(url);host=p.hostname or '';parts=p.path.strip('/').split('/');v=''
        if p.scheme not in ('https','http'):return None
        if host=='youtu.be':v=parts[0]
        elif host in ('youtube.com','www.youtube.com','m.youtube.com'):
            if p.path=='/watch':v=parse_qs(p.query).get('v',[''])[0]
            elif len(parts)>1 and parts[0] in ('shorts','live','embed'):v=parts[1]
        return v if re.fullmatch(r'[A-Za-z0-9_-]{11}',v) else None
    except Exception:return None
@app.get('/api/diagnostics')
def diagnostics():
    free=shutil.disk_usage(ROOT).free;reserve=min(512*1024**2,int(free*.1));limit=max(0,min(LIMIT,free-reserve))
    return {'configured_limit':LIMIT,'available_upload_bytes':limit,'free_bytes':free,'chunk_bytes':CHUNK,'ffmpeg':bool(shutil.which('ffmpeg')),'ffprobe':bool(shutil.which('ffprobe')),'asr_configured':bool(os.getenv('ZENVYRA_WHISPER_MODEL')),'deployment':'single-owner protected hosting' if os.getenv('ZENVYRA_HOSTED','').lower() in ('1','true','yes') else 'single-user local','youtube':'reference + authorized original file only'}
@app.get('/api/projects')
def projects():
    with db() as c:return [decoded(r) for r in c.execute('SELECT * FROM projects ORDER BY created DESC')]
@app.post('/api/projects')
def create(p:Project):
    if p.workspace not in ('gallery','youtube'):raise HTTPException(422,'Invalid workspace')
    if p.workspace=='youtube' and not youtube_id(p.url or ''):raise HTTPException(422,'Valid YouTube video, Shorts or live URL required')
    if p.workspace=='gallery' and p.url:raise HTTPException(422,'Gallery cannot have YouTube URL')
    id=uuid.uuid4().hex;directory(id).mkdir()
    with db() as c:c.execute('INSERT INTO projects(id,workspace,name,url,created) VALUES(?,?,?,?,?)',(id,p.workspace,p.name,p.url,time.time()))
    return decoded(row(id))
@app.get('/api/projects/{id}')
def get(id:str):
    r=decoded(row(id))
    with db() as c:r['jobs']=[decoded(j) for j in c.execute('SELECT * FROM jobs WHERE project=? ORDER BY created DESC',(id,))]
    return r
@app.put('/api/projects/{id}/settings')
def save(id:str,s:dict):row(id);patch(id,settings=json.dumps(valid(s)));return {'ok':True}
@app.post('/api/projects/{id}/upload')
def begin(id:str,b:Begin):
    with lock:
        r=row(id)
        if r['fingerprint']:
            if r['fingerprint']!=b.fingerprint or r['size']!=b.size:raise HTTPException(409,'Select the same file to resume, or create a new project')
            return {'received':r['received'],'complete':bool(r['meta'])}
        if b.size>diagnostics()['available_upload_bytes']:raise HTTPException(413,'File exceeds available storage limit')
        src(id).touch();patch(id,size=b.size,fingerprint=b.fingerprint,received=0);return {'received':0,'complete':False}
@app.put('/api/projects/{id}/upload/{offset}')
async def chunk(id:str,offset:int,request:Request):
    body=bytearray()
    async for part in request.stream():
        body.extend(part)
        if len(body)>CHUNK:raise HTTPException(413,'Chunk too large')
    if not body:raise HTTPException(400,'Empty chunk')
    with lock:
        r=row(id)
        if r['meta'] or not r['fingerprint'] or offset!=r['received'] or offset+len(body)>r['size']:raise HTTPException(409,'Upload offset mismatch; resume from server offset')
        if len(body)>shutil.disk_usage(ROOT).free:raise HTTPException(507,'Insufficient disk space')
        with src(id).open('r+b') as f:f.seek(offset);f.write(body);f.flush();os.fsync(f.fileno());f.truncate(offset+len(body))
        received=offset+len(body);patch(id,received=received);return {'received':received}
@app.post('/api/projects/{id}/complete')
def complete(id:str):
    with lock:
        r=row(id)
        if r['meta']:return json.loads(r['meta'])
        if not r['size'] or r['received']!=r['size'] or src(id).stat().st_size!=r['size']:raise HTTPException(409,'Upload incomplete')
        try:m=engine.probe(src(id))
        except Exception as e:raise HTTPException(422,'Invalid media: '+str(e)[:250])
        patch(id,meta=json.dumps(m));return m
@app.get('/api/projects/{id}/source')
def source(id:str):
    r=row(id)
    if not r['meta']:raise HTTPException(409,'Source not ready')
    m=json.loads(r['meta']); fmt=m.get('format',''); media='video/webm' if 'webm' in fmt and m['codec'] in ('vp8','vp9','av1') else 'video/mp4' if 'mp4' in fmt or 'mov' in fmt else 'application/octet-stream'
    return FileResponse(src(id),media_type=media)
@app.post('/api/projects/{id}/analyze')
def analyze(id:str,s:dict):
    r=row(id);s=valid(s)
    if not r['meta']:raise HTTPException(409,'Complete source upload first')
    patch(id,settings=json.dumps(s));return enqueue(id,'analyze',s)
@app.post('/api/projects/{id}/export')
def export(id:str,p:dict):
    r=row(id);s=valid(p.get('settings',{}));m=json.loads(r['meta']) if r['meta'] else None
    if not m:raise HTTPException(409,'Source not ready')
    if (s.get('tracking') or s.get('border')) and not r['analysis']:raise HTTPException(409,'Analyze source before tracking or border detection')
    try:
        clip=p['clip'];a,b=float(clip['start']),float(clip['end']);
        if not 0<=a<b<=m['duration']+.001:raise ValueError('Invalid clip interval')
        caps=[]
        for c in clip.get('captions',[])[:1000]:
            ca,cb=float(c['start']),float(c['end']);
            if not 0<=ca<cb<=m['duration']+.001:raise ValueError('Invalid caption interval')
            caps.append({'start':ca,'end':cb,'text':str(c['text'])[:2000]})
        clip={'start':a,'end':b,'captions':caps};ratios=s.get('ratios',['9:16']);jobs=[]
        for ratio in ratios:engine.geometry(m,ratio,s)
    except (KeyError,ValueError,TypeError,ZeroDivisionError,OverflowError) as e:raise HTTPException(422,str(e))
    patch(id,settings=json.dumps(s))
    render_settings={k:s[k] for k in ('quality','mode','tracking','profile','border','burn_captions','x','y','ratio_w','ratio_h') if k in s}
    for ratio in ratios:jobs.append(enqueue(id,'export',{'clip':clip,'ratio':ratio,'settings':render_settings}))
    return {'jobs':jobs}
@app.get('/api/projects/{id}/exports/{name}')
def download(id:str,name:str,download:bool=False):
    row(id)
    if not re.fullmatch(r'[a-f0-9]{24}\.mp4',name):raise HTTPException(404)
    f=directory(id)/name
    if not f.is_file():raise HTTPException(404)
    return FileResponse(f,media_type='video/mp4',filename='Zenvyra-'+name if download else None)
@app.get('/')
def home():return FileResponse(Path(__file__).parent/'static/index.html')
app.mount('/static',StaticFiles(directory=Path(__file__).parent/'static'),name='static')
