"""Local media engine. No network calls or credentials."""
import json, math, os, subprocess, time, re
from pathlib import Path
import cv2
import numpy as np

RATIOS={'9:16':9/16,'16:9':16/9,'4:3':4/3,'3:4':3/4,'1:1':1,'4:5':4/5,'21:9':21/9}
def run(args):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace')[-1600:])
    return p.stdout

def probe(path):
    d=json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
    v=next((s for s in d['streams'] if s['codec_type']=='video' and not s.get('disposition',{}).get('attached_pic')),None)
    if not v: raise ValueError('No video stream')
    dur=float(d['format'].get('duration') or v.get('duration') or 0)
    if not math.isfinite(dur) or dur<=0: raise ValueError('Invalid video duration')
    w,h=v['width'],v['height']; rot=next((s.get('rotation',0) for s in v.get('side_data_list',[]) if 'rotation' in s),0)
    if abs(int(rot))%180==90: w,h=h,w
    return dict(width=w,height=h,duration=dur,audio=any(s['codec_type']=='audio' for s in d['streams']),codec=v['codec_name'],sar=v.get('sample_aspect_ratio','1:1'),rotation=rot,format=d['format'].get('format_name',''))

def even(n): return max(2,int(n)//2*2)
def geometry(meta,ratio,settings,border=None):
    w,h=meta['width'],meta['height']; x,y=0,0
    if settings.get('border') and border: x,y,w,h=border
    w,h=even(w),even(h)
    if ratio=='Original': r=w/h
    elif ratio=='Custom':
        r=float(settings.get('ratio_w',0))/float(settings.get('ratio_h',0))
    else: r=RATIOS[ratio]
    if not 0.2<=r<=5: raise ValueError('Aspect ratio must be between 1:5 and 5:1')
    mode=settings.get('mode','fit'); q=settings.get('quality','Source Max')
    longmax={'720p':1280,'1080p':1920,'4K':3840}.get(q,max(w,h))
    if r>=1: ow,oh=longmax,longmax/r
    else: ow,oh=longmax*r,longmax
    if q!='Source Max':
        shortmax={'720p':720,'1080p':1080,'4K':2160}[q]; scale=min(1,shortmax/min(ow,oh));ow*=scale;oh*=scale
    elif mode=='fill':
        scale=min(1,w/ow,h/oh);ow*=scale;oh*=scale
    if q=='Source Max' and ratio=='Original': ow,oh=w,h
    from fractions import Fraction
    if ratio!='Original':
        f=Fraction(r).limit_denominator(10000); unitw,unith=2*f.numerator,2*f.denominator
        multiple=int(min(ow/unitw,oh/unith))
        if multiple>=1:ow,oh=multiple*unitw,multiple*unith
    ow,oh=even(ow),even(oh)
    if max(ow,oh)>7680 or ow*oh>40_000_000: raise ValueError('Output resolution exceeds local encoder guard')
    cw,ch=w,h
    if mode=='fill':
        if w/h>ow/oh: cw=even(h*ow/oh)
        else: ch=even(w*oh/ow)
    px=max(0,min(1,float(settings.get('x',.5)))); py=max(0,min(1,float(settings.get('y',.5))))
    cx=int((w-cw)*px)//2*2; cy=int((h-ch)*py)//2*2
    return dict(base=[x,y,w,h],crop=[cx,cy,cw,ch],output=[ow,oh],mode=mode)

def detect_border(frames,w,h):
    if len(frames)<3:return None
    masks=[np.max(f,axis=2)<=12 for f in frames]
    allblack=np.logical_and.reduce(masks); sh,sw=allblack.shape
    rows=allblack.mean(1)>=.998; cols=allblack.mean(0)>=.998
    def count(a):
        i=0
        while i<len(a) and a[i]:i+=1
        return i
    t,b,l,r=count(rows),count(rows[::-1]),count(cols),count(cols[::-1])
    if t+b>sh*.35 or l+r>sw*.35:return None
    if min(t,b)<2 or abs(t-b)>max(2,.1*max(t,b)):t=b=0
    if min(l,r)<2 or abs(l-r)>max(2,.1*max(l,r)):l=r=0
    if not any((t,b,l,r)):return None
    x=int(l*w/sw)//2*2;y=int(t*h/sh)//2*2
    right=int(r*w/sw)//2*2;bottom=int(b*h/sh)//2*2
    return [x,y,even(w-x-right),even(h-y-bottom)]

def analyze(path,meta,progress=lambda *_:None):
    """One-pass sampled vision + audio energy, optional local Whisper. Real measurements, not semantic AI claims."""
    cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS) or 25
    duration=meta['duration']; step=max(.5,duration/1800); samples=[]; prev=None; frames=[]
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    t=0.; idx=0
    try:
        while t<duration:
            cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,f=cap.read()
            if not ok: break
            small=cv2.resize(f,(192,108)); gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
            diff=np.abs(gray.astype(np.float32)-prev) if prev is not None else np.zeros_like(gray,dtype=np.float32)
            motion=float(diff.mean()/255); hist=cv2.calcHist([gray],[0],None,[32],[0,256]); hist=cv2.normalize(hist,hist,alpha=1,norm_type=cv2.NORM_L1)
            scene=float(cv2.compareHist(last_hist,hist,cv2.HISTCMP_BHATTACHARYYA)) if prev is not None else 0.
            yy,xx=np.indices(diff.shape); total=diff.sum(); mx=float((xx*diff).sum()/total/191) if total>50 else .5; my=float((yy*diff).sum()/total/107) if total>50 else .5
            faces=detector.detectMultiScale(gray,1.15,3,minSize=(16,16)) if not detector.empty() else []
            face=None
            if len(faces):
                a,b,c,d=max(faces,key=lambda z:z[2]*z[3]); face=[float((a+c/2)/192),float((b+d/2)/108)]
            samples.append(dict(t=round(t,3),motion=motion,scene=scene,activity=[mx,my],face=face,audio=0.))
            if idx%max(1,int(duration/step/60))==0:frames.append(small)
            prev=gray.astype(np.float32);last_hist=hist;t+=step;idx+=1
            if idx%20==0: progress(min(.75,t/duration*.75),'Sampling actual video frames')
    finally:cap.release()
    if not samples:raise ValueError('OpenCV could not decode source frames')
    if meta['audio']:
        p=subprocess.Popen(['ffmpeg','-v','error','-i',str(path),'-vn','-ac','1','-ar','8000','-f','s16le','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        try:
            for s in samples:
                data=p.stdout.read(max(2,int(step*8000)*2)); a=np.frombuffer(data[:len(data)//2*2],dtype='<i2').astype(np.float32)/32768
                s['audio']=float(np.sqrt(np.mean(a*a))) if len(a) else 0.
        finally:
            p.stdout.close();p.wait(timeout=30)
    captions=[]; asr='not configured'
    model_path=os.getenv('ZENVYRA_WHISPER_MODEL','')
    if model_path:
        try:
            from faster_whisper import WhisperModel
            if not Path(model_path).is_dir(): raise ValueError('Use a locally installed model directory')
            progress(.83,'Local speech recognition')
            model=WhisperModel(model_path,device='cpu',compute_type='int8',local_files_only=True)
            segments,_=model.transcribe(str(path),vad_filter=True)
            captions=[dict(start=float(s.start),end=float(s.end),text=s.text.strip()) for s in segments if s.text.strip()];asr='local Whisper'
        except Exception as e:asr='unavailable: '+str(e)[:200]
    progress(.92,'Checking black-border candidate')
    candidate=detect_border(frames,meta['width'],meta['height'])
    if candidate:
        c=cv2.VideoCapture(str(path)); valid=True; n=0; bx,by,bw,bh=candidate
        try:
            while True:
                ok,fr=c.read()
                if not ok:break
                fh,fw=fr.shape[:2]
                if (fw,fh)!=(meta['width'],meta['height']):valid=False;break
                strips=[fr[:by],fr[by+bh:],fr[:,:bx],fr[:,bx+bw:]]
                if any(strip.size and int(strip.max())>12 for strip in strips):valid=False;break
                n+=1
        finally:c.release()
        expected=int(duration*fps)
        if not valid or n<max(1,expected-2):candidate=None
    progress(.95,'Saving measurements')
    return dict(samples=samples,border=candidate,captions=captions,asr=asr,method='sampled visual motion, histogram changes, audio RMS, optional local ASR',step=step)

def highlights(analysis,meta,settings):
    duration=meta['duration']; sec=float(settings.get('seconds',30));sec=min(duration,sec)
    minsec=min(sec,float(settings.get('min_seconds',sec))); lengths=sorted(set([minsec,(minsec+sec)/2,sec]))
    if sec<=0:raise ValueError('Duration must be positive')
    count=settings.get('count','Auto'); count=min(20,max(1,math.ceil(duration/sec/2))) if count=='Auto' else int(count)
    if not 1<=count<=100:raise ValueError('Clip count must be 1–100')
    samples=analysis['samples']; ts=np.array([s['t'] for s in samples]);
    motion=np.array([s['motion'] for s in samples]); scene=np.array([s['scene'] for s in samples]); audio=np.array([s['audio'] for s in samples]);
    def normalized(a):return a/(max(float(np.percentile(a,95)),.0001))
    game=settings.get('preset')=='gameplay'; signal=(.65 if game else .45)*normalized(motion)+.2*normalized(scene)+(.15 if game else .35)*normalized(audio)
    words=set(re.findall(r'\w+',settings.get('topic','').lower())); words-= {'the','a','an','and','of','in','to'}
    for c in analysis['captions']:
        hits=words.intersection(re.findall(r'\w+',c['text'].lower()))
        if hits:signal[(ts>=c['start'])&(ts<=c['end'])]+=min(2,len(hits))
    candidates=[]
    for length in lengths:
        starts=np.arange(0,max(0,duration-length)+.00001,max(1,length/4)).tolist();starts.append(max(0,duration-length))
        for start in sorted(set(starts)):
            mask=(ts>=start)&(ts<start+length)
            if not mask.any():continue
            value=float(signal[mask].mean()); candidates.append((value,start,length,float(motion[mask].mean()),float(audio[mask].mean()),float(scene[mask].mean())))
    if not candidates:raise ValueError('No measurable candidate windows')
    selected=[]
    for value,start,length,m,a,s in sorted(candidates,reverse=True):
        end=min(duration,start+length)
        if any(max(0,min(end,z['end'])-max(start,z['start']))>.4*min(length,z['end']-z['start']) for z in selected):continue
        caps=[dict(start=max(start,c['start']),end=min(end,c['end']),text=c['text']) for c in analysis['captions'] if c['end']>start and c['start']<end]
        selected.append(dict(start=round(start,3),end=round(end,3),reason=f'Measured motion {m:.3f} · audio RMS {a:.3f} · scene change {s:.3f}',captions=caps))
        if len(selected)>=count:break
    return selected

def track_expression(samples,start,end,axis,profile,dimension,crop,base_offset,original,default):
    if dimension<=crop:return '0'
    points=[];smooth=None
    for s in samples:
        if not start<=s['t']<=end:continue
        loc=s['face'] if profile in ('speaker','auto') and s['face'] else (None if profile=='speaker' else s['activity'])
        if loc is None:continue
        p=max(0,min(dimension-crop,loc[axis]*original-base_offset-crop/2));smooth=p if smooth is None else .3*p+.7*smooth
        points.append((s['t']-start,int(smooth)//2*2))
    if not points:return str(default)
    stride=max(1,math.ceil(len(points)/60));points=points[::stride];expr=str(points[-1][1])
    for (t0,p0),(t1,p1) in reversed(list(zip(points,points[1:]))):
        if t1>t0:expr=f'if(lt(t,{t1:.4f}),{p0}+({p1-p0})*clip((t-{t0:.4f})/{t1-t0:.4f},0,1),{expr})'
    return expr

def export(path,out,meta,analysis,clip,ratio,settings,progress=lambda *_:None):
    start,end=float(clip['start']),float(clip['end']); dur=end-start
    if not 0<=start<end<=meta['duration']+.001:raise ValueError('Invalid clip interval')
    g=geometry(meta,ratio,settings,analysis.get('border')); bx,by,w,h=g['base']; cx,cy,cw,ch=g['crop']; ow,oh=g['output'];mode=g['mode']
    vf=f'crop={w}:{h}:{bx}:{by},setsar=1'; complex_filter=False
    if mode=='fill':
        ex,ey=str(cx),str(cy)
        if settings.get('tracking'):
            prof=settings.get('profile','auto');ex=track_expression(analysis['samples'],start,end,0,prof,w,cw,bx,meta['width'],cx);ey=track_expression(analysis['samples'],start,end,1,prof,h,ch,by,meta['height'],cy)
        vf+=f",crop={cw}:{ch}:'{ex}':'{ey}',scale={ow}:{oh},setsar=1"
    elif mode=='blur':
        vf=f'[0:v]{vf},split[bg][fg];[bg]scale={ow}:{oh}:force_original_aspect_ratio=increase:force_divisible_by=2,crop={ow}:{oh},boxblur=12:2[b];[fg]scale={ow}:{oh}:force_original_aspect_ratio=decrease:force_divisible_by=2[f];[b][f]overlay=(W-w)/2:(H-h)/2,setsar=1';complex_filter=True
    else:vf+=f',scale={ow}:{oh}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black,setsar=1'
    out=Path(out); tmp=out.with_suffix('.part.mp4'); subtitle=out.with_suffix('.srt')
    def srt_time(s):
        ms=max(0,round(s*1000));return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
    try:
        if settings.get('burn_captions') and clip.get('captions'):
            lines=[]
            for c in clip['captions']:
                a,b=max(start,float(c['start'])),min(end,float(c['end']))
                if b>a:lines.append(f'{len(lines)+1}\n{srt_time(a-start)} --> {srt_time(b-start)}\n{str(c["text"]).replace(chr(0),"")[:2000]}\n')
            if lines:
                subtitle.write_text('\n'.join(lines),encoding='utf8'); vf+=f",subtitles=filename='{subtitle.name}'"
        args=['ffmpeg','-y','-v','error','-ss',str(start),'-i',str(Path(path).resolve()),'-t',str(dur),'-filter_threads','1','-filter_complex_threads','1']
        if complex_filter:args+=['-filter_complex',vf+'[out]','-map','[out]']
        else:args+=['-vf',vf,'-map','0:v:0']
        args+=['-map','0:a:0?','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-threads','2','-c:a','aac','-b:a','160k','-movflags','+faststart','-progress','pipe:1','-nostats',str(tmp.resolve())]
        import tempfile
        with tempfile.TemporaryFile() as errors:
            p=subprocess.Popen(args,cwd=out.parent,stdout=subprocess.PIPE,stderr=errors,text=False)
            for line in p.stdout:
                if line.startswith(b'out_time_us='):
                    try:progress(min(.98,int(line.split(b'=')[1])/1e6/dur),'Encoding MP4')
                    except ValueError:pass
            if p.wait()!=0:
                errors.seek(0);raise RuntimeError(errors.read().decode(errors='replace')[-1600:])
        result=probe(tmp)
        if (result['width'],result['height'])!=(ow,oh) or result['sar'] not in ('1:1','N/A') or abs(result['duration']-dur)>.35:raise RuntimeError('Export verification failed: geometry or duration mismatch')
        os.replace(tmp,out);return result
    finally:
        tmp.unlink(missing_ok=True);subtitle.unlink(missing_ok=True)
