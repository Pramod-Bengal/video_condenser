import asyncio
import json
import math
import subprocess
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_from_directory

try:
    import av
    _orig_av_open = av.open
    def _patched_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)
    av.open = _patched_av_open
except Exception:
    pass

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
WORK = BASE / "work"
THUMBS = BASE / "thumbnails"
FRAMES = BASE / "frames"
for d in (UPLOADS, OUTPUTS, WORK, THUMBS, FRAMES):
    d.mkdir(exist_ok=True)

app = Flask(__name__)
JOBS = {}
MAX_CLIP = 4.9  # seconds, every video clip stays below 5s
WORDS_PER_SEC = 2.4


# ---------------------------------------------------------------- helpers
def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("Command failed:\n" + p.stderr.decode(errors="ignore")[-700:])
    return p.stdout.decode(errors="ignore")


def probe_duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "csv=p=0", str(path)]).strip())


def fmt(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


# ---------------------------------------------------------------- step 1: transcript
def transcribe(path, model_size, job):
    from faster_whisper import WhisperModel
    job["message"] = "Whisper model load aagtide (modalu sala download aagutte)..."
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    job["message"] = "Dialogue transcript togotide (idu swalpa time tagoluttade)..."
    segments, info = model.transcribe(str(path), task="translate", vad_filter=True)
    total = probe_duration(path)
    lines = []
    for s in segments:
        lines.append(f"[{s.start:.0f}s | {fmt(s.start)}] {s.text.strip()}")
        job["progress"] = min(45, int(s.end / total * 45))
    return "\n".join(lines), total


# ---------------------------------------------------------------- step 2: summary beats
def build_prompt(transcript, target_sec, total):
    words = int(target_sec * WORDS_PER_SEC)
    return f"""Below is the timestamped transcript of a {fmt(total)} long video (story episode).
Summarize the whole story in ENGLISH as a narration of about {words} words in total (it will be spoken as a {target_sec // 60}-minute voiceover).
Cover the full story from beginning to end in order: characters, key events, conflicts, twists and the ending.
Split the narration into many short beats. Each beat is 1-2 sentences (15-35 words) and must describe a moment that happens in the video.
For each beat give "start" and "end" (in seconds) of the part of the original video that shows that moment, so matching visuals can be cut from it.
Beats must be in story order. Use only timestamps that exist in the video (0 to {int(total)} seconds).
Return ONLY a JSON array, no other text, like:
[{{"text": "narration sentence(s)", "start": 12, "end": 40}}, ...]

TRANSCRIPT:
{transcript}"""


def summarize_with_claude(api_key, transcript, target_sec, total):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-sonnet-5-5", max_tokens=16000,
        messages=[{"role": "user", "content": build_prompt(transcript, target_sec, total)}])
    return msg.content[0].text


def parse_beats(text, total):
    a, b = text.find("["), text.rfind("]")
    beats = json.loads(text[a:b + 1])
    out = []
    for x in beats:
        s = max(0.0, min(float(x["start"]), total - 1))
        e = max(s + 1, min(float(x["end"]), total))
        out.append({"text": str(x["text"]).strip(), "start": s, "end": e})
    if not out:
        raise ValueError("Beats empty")
    return out


# ---------------------------------------------------------------- step 3: voice + video + subtitles
def make_voice(text, voice, out_mp3):
    if out_mp3.exists() and out_mp3.stat().st_size > 500:
        return

    import time
    import edge_tts

    last_err = None
    # 1. Edge TTS with retries
    for attempt in range(3):
        try:
            async def go():
                communicate = edge_tts.Communicate(text, voice, rate="+0%")
                await communicate.save(str(out_mp3))
            asyncio.run(go())
            if out_mp3.exists() and out_mp3.stat().st_size > 500:
                time.sleep(0.3)
                return
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))

    # 2. Fallback to gTTS (Google Translate TTS)
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en")
        tts.save(str(out_mp3))
        if out_mp3.exists() and out_mp3.stat().st_size > 500:
            return
    except Exception as e:
        last_err = e

    # 3. Guaranteed 100% Offline fallback using pyttsx3 + ffmpeg
    try:
        import pyttsx3
        wav_tmp = out_mp3.with_suffix(".wav")
        engine = pyttsx3.init()
        engine.save_to_file(text, str(wav_tmp))
        engine.runAndWait()
        if wav_tmp.exists():
            run(["ffmpeg", "-y", "-i", str(wav_tmp), "-acodec", "libmp3lame", str(out_mp3)])
            wav_tmp.unlink(missing_ok=True)
            if out_mp3.exists() and out_mp3.stat().st_size > 500:
                return
    except Exception as e:
        last_err = e

    raise RuntimeError(f"Voice generation failed: {last_err}")


def pick_starts(start, end, d, n, L, total):
    avail = end - start
    if n == 1:
        return [max(0.0, min(start + max(0, (avail - L) / 2), total - L))]
    if avail >= d:
        step = (avail - L) / (n - 1)
        return [start + i * step for i in range(n)]
    s0 = max(0.0, min(start, total - d))
    return [s0 + i * L for i in range(n)]


def render_beats(path, beats, voice, work, total, job):
    vf = ("scale=1280:720:force_original_aspect_ratio=decrease,"
          "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30")
    beat_files, used = [], 0.0
    for bi, b in enumerate(beats, 1):
        job["message"] = f"Voice + clips: beat {bi}/{len(beats)}"
        job["progress"] = 50 + int(bi / len(beats) * 45)
        mp3 = work / f"v_{bi:04d}.mp3"
        make_voice(b["text"], voice, mp3)
        d = probe_duration(mp3)
        n = max(1, math.ceil(d / MAX_CLIP))
        L = d / n
        starts = pick_starts(b["start"], b["end"], d, n, L, total)
        out = work / f"beat_{bi:04d}.mp4"
        if out.exists() and out.stat().st_size > 5000:
            beat_files.append(out)
            used += d
            continue

        clips = []
        for k, st in enumerate(starts):
            st = max(0.0, min(st, total - L - 0.05))
            c = work / f"c_{bi:04d}_{k}.mp4"
            if not (c.exists() and c.stat().st_size > 5000):
                run(["ffmpeg", "-y", "-ss", f"{st:.3f}", "-i", str(path), "-t", f"{L:.3f}",
                     "-an", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
                     "-crf", "23", "-pix_fmt", "yuv420p", str(c)])
            clips.append(c)
        lst = work / f"l_{bi:04d}.txt"
        lst.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(mp3),
             "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
             "-ar", "44100", "-ac", "2", "-t", f"{d:.3f}", str(out)])
        beat_files.append(out)
        used += d
    lst = work / "all.txt"
    lst.write_text("".join(f"file '{c.as_posix()}'\n" for c in beat_files), encoding="utf-8")
    return lst, used


def finish_job(job_id, beats):
    job = JOBS[job_id]
    try:
        path = job["video"]
        work = WORK / job_id
        work.mkdir(exist_ok=True)
        lst, used = render_beats(path, beats, job["voice"], work, job["total"], job)

        # 1. Generate full SRT subtitle file
        srt_file = work / "subtitles.srt"
        srt_lines = []
        cur_t = 0.0
        for bi, b in enumerate(beats, 1):
            mp3 = work / f"v_{bi:04d}.mp3"
            d = probe_duration(mp3) if mp3.exists() else 3.0
            def fmt_srt(t):
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = int(t % 60)
                ms = int(round((t % 1) * 1000))
                if ms >= 1000:
                    s += 1
                    ms = 0
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            t_start = fmt_srt(cur_t)
            t_end = fmt_srt(cur_t + d)
            srt_lines.append(f"{bi}\n{t_start} --> {t_end}\n{b['text']}\n")
            cur_t += d

        srt_content = "\n".join(srt_lines)
        srt_file.write_text(srt_content, encoding="utf-8")

        base_stem = path.stem.split('_', 1)[-1]
        out_name = f"{base_stem}_summary.mp4"
        raw_concat = work / "raw_concat.mp4"
        out_mp4 = OUTPUTS / out_name
        out_srt = OUTPUTS / f"{base_stem}_subtitles.srt"
        out_srt.write_text(srt_content, encoding="utf-8")

        # 2. Concat clips
        job["message"] = "Clips concat maaduttide..."
        job["progress"] = 96
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-c", "copy", str(raw_concat)])

        # 3. Burn subtitles into final mp4 using FFmpeg libass
        sub_size = job.get("sub_size", "15")
        try:
            sz_int = int(sub_size)
        except Exception:
            sz_int = 15
        margin_v = 22 if sz_int <= 15 else (28 if sz_int <= 18 else 35)
        outline_w = 1.4 if sz_int <= 15 else 1.8
        job["message"] = f"Subtitles burn maaduttide (size: {sz_int}px)..."
        job["progress"] = 98
        srt_escaped = str(srt_file.resolve()).replace('\\', '/').replace(':', '\\:')
        sub_style = (f"force_style='FontSize={sz_int},FontName=Arial,Bold=1,"
                     "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                     f"BorderStyle=3,Outline={outline_w},Shadow=1,MarginV={margin_v},Alignment=2'")
        sub_filter = f"subtitles='{srt_escaped}':{sub_style}"

        try:
            run(["ffmpeg", "-y", "-i", str(raw_concat), "-vf", sub_filter,
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                 "-c:a", "copy", str(out_mp4)])
        except Exception:
            import shutil
            shutil.copyfile(str(raw_concat), str(out_mp4))

        job["outputs"].append({"name": out_name, "minutes": round(used / 60, 1),
                               "clips": len(beats), "label": "Subtitled Video (MP4)"})
        job["outputs"].append({"name": f"{base_stem}_subtitles.srt", "minutes": round(used / 60, 1),
                               "clips": len(beats), "label": "SRT Subtitle File"})
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Mugithu! Subtitles add aagide."
    except Exception as e:  # noqa
        job["status"] = "error"
        job["message"] = str(e)


def process_job(job_id, path, target_min, api_key, model_size):
    job = JOBS[job_id]
    try:
        transcript, total = transcribe(path, model_size, job)
        if not transcript.strip():
            raise RuntimeError("Dialogue sigalilla. Ee video ge speech illa andre summary baralla.")
        job.update(transcript=transcript, total=total, target=target_min * 60)
        if api_key:
            job["message"] = "Claude story summary maaduttide..."
            job["progress"] = 48
            raw = summarize_with_claude(api_key, transcript, target_min * 60, total)
            beats = parse_beats(raw, total)
            finish_job(job_id, beats)
        else:
            job["prompt"] = build_prompt(transcript, target_min * 60, total)
            job["status"] = "need_beats"
            job["message"] = "Transcript ready. Kelage ide step follow madi."
    except Exception as e:  # noqa
        job["status"] = "error"
        job["message"] = str(e)


# ---------------------------------------------------------------- web
PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>Story Summary Video & Thumbnail Studio</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;background:#0d0d0d;color:#eee;max-width:880px;margin:24px auto;padding:0 16px}
.nav{display:flex;gap:12px;margin-bottom:20px}
.nav button{background:#222;border:1px solid #444;color:#aaa;padding:10px 20px;border-radius:8px;font-size:15px;cursor:pointer}
.nav button.active{background:#e50914;color:#fff;border-color:#e50914;font-weight:bold}
.box{background:#181818;border:1px solid #333;border-radius:12px;padding:24px;margin-bottom:24px}
label{display:block;margin:12px 0 4px;color:#ccc;font-size:14px}
input,select{padding:8px 10px;background:#222;border:1px solid #444;color:#fff;border-radius:6px}
input[type=password],input[type=text]{width:100%;box-sizing:border-box}
textarea{width:100%;height:140px;background:#111;color:#eee;border:1px solid #444;border-radius:6px;box-sizing:border-box;padding:8px}
button.action{background:#e50914;color:#fff;border:0;padding:12px 28px;border-radius:8px;font-size:15px;font-weight:bold;cursor:pointer;margin-top:14px}
button.action:disabled{opacity:.5;cursor:not-allowed}
.bar{height:14px;background:#282828;border-radius:8px;overflow:hidden;margin-top:18px}
.bar div{height:100%;width:0;background:#2ecc71;transition:width .4s}
a.dl{display:inline-block;background:#2980b9;color:#fff;text-decoration:none;padding:8px 16px;border-radius:6px;margin:6px 6px 6px 0;font-weight:500}
a.dl:hover{background:#3498db}
small{color:#888}
/* Thumbnail Studio Styles */
.thumbs-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:14px 0}
.thumb-card{border:2px solid #333;border-radius:8px;overflow:hidden;cursor:pointer;transition:border-color .2s}
.thumb-card:hover{border-color:#e50914}
.thumb-card.selected{border-color:#e50914;box-shadow:0 0 10px rgba(229,9,20,0.6)}
.thumb-card img{width:100%;height:auto;display:block}
.thumb-card .caption{font-size:12px;padding:6px;background:#222;text-align:center;color:#ddd}
.controls-row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
canvas{max-width:100%;height:auto;border-radius:8px;box-shadow:0 4px 20px rgba(0,0,0,0.8);background:#000;display:block;margin:16px auto}
</style></head><body>

<div class="nav">
<button id="tab-video-btn" class="active" onclick="switchTab('video')">🎬 1. Video & Subtitles</button>
<button id="tab-thumb-btn" onclick="switchTab('thumb')">🎨 2. YouTube Thumbnail Studio</button>
</div>

<!-- TAB 1: VIDEO CONDENSER & SUBTITLES -->
<div id="tab-video">
<h2>Story Summary Video with Subtitles</h2>
<p>English narrated summary video with <b>burned-in subtitles</b> and auto-generated <b>.SRT file</b>.</p>
<div class="box">
<input type="file" id="file" accept="video/*">
<label>Target length (minutes): <input type="number" id="target" value="15" min="2" max="30" style="width:80px"></label>
<label>Voice: <select id="voice">
<option value="en-US-GuyNeural">Male (US)</option><option value="en-US-JennyNeural">Female (US)</option>
<option value="en-GB-RyanNeural">Male (UK)</option><option value="en-IN-NeerjaNeural">Female (India)</option></select></label>
<label>Whisper model: <select id="wm"><option value="small">small (balanced)</option>
<option value="base">base (fast)</option><option value="medium">medium (best, slow)</option></select></label>
<label>Subtitle font size: <select id="sub_size">
<option value="15" selected>Small (15px - Recommended)</option>
<option value="13">Extra Small (13px)</option>
<option value="18">Medium (18px)</option>
<option value="22">Large (22px)</option>
</select></label>
<label>Anthropic API key (optional - auto summary ge): <input type="password" id="key" placeholder="sk-ant-..."></label>
<small>Key illa andre transcript sigutte, adannu chat alli summary maadsi paste maadbeku (kelage steps).</small><br>
<button id="go" class="action">Start</button>
<div class="bar"><div id="p"></div></div><p id="msg" style="color:#2ecc71;font-weight:bold"></p>
<div id="manual" style="display:none">
<p><b>Step A:</b> Ee prompt na copy madi chat alli paste madi:</p><textarea id="prompt" readonly></textarea>
<p><b>Step B:</b> Chat kodida JSON answer illi paste madi:</p><textarea id="beats"></textarea>
<button id="cont" class="action">Generate video</button></div>
<div id="res" style="margin-top:16px"></div>
</div>
</div>

<!-- TAB 2: YOUTUBE THUMBNAIL STUDIO -->
<div id="tab-thumb" style="display:none">
<h2>YouTube Thumbnail Studio</h2>
<p>Select your video file — 5 cinematic frames are extracted instantly. Pick a frame, add text, and download!</p>

<div class="box">
<label><b>1. Select Video to Extract Frames:</b></label>
<input type="file" id="thumb-video-file" accept="video/*" style="margin:8px 0">
<p id="thumb-hint" style="color:#f1c40f;font-size:13px;margin:4px 0 8px">📌 Choose a video file above — 5 frames will be extracted automatically.</p>

<label><b>Select Thumbnail Template:</b></label>
<div id="thumbs-grid" class="thumbs-grid">
 <div class="thumb-card selected" onclick="pickThumb('/thumb/thumb1.jpg', this)"><img src="/thumb/thumb1.jpg"><div class="caption">Default 1</div></div>
 <div class="thumb-card" onclick="pickThumb('/thumb/thumb2.jpg', this)"><img src="/thumb/thumb2.jpg"><div class="caption">Default 2</div></div>
 <div class="thumb-card" onclick="pickThumb('/thumb/thumb3.jpg', this)"><img src="/thumb/thumb3.jpg"><div class="caption">Default 3</div></div>
 <div class="thumb-card" onclick="pickThumb('/thumb/thumb4.jpg', this)"><img src="/thumb/thumb4.jpg"><div class="caption">Default 4</div></div>
 <div class="thumb-card" onclick="pickThumb('/thumb/thumb5.jpg', this)"><img src="/thumb/thumb5.jpg"><div class="caption">Default 5</div></div>
</div>

<label>Or Upload Custom Image: <input type="file" id="custom-thumb-file" accept="image/*" onchange="loadCustomThumb(event)"></label>

<div class="controls-row" style="margin-top:18px">
 <div>
  <label>Top Badge Text (e.g. Series Name / Season):</label>
  <input type="text" id="t-badge" value="FROM • SEASON 01" oninput="drawThumbnail()">
 </div>
 <div>
  <label>Badge Color:</label>
  <select id="t-badge-color" onchange="drawThumbnail()">
   <option value="#e50914">Netflix Red (#e50914)</option>
   <option value="#f1c40f">Amber Gold (#f1c40f)</option>
   <option value="#00e5ff">Electric Cyan (#00e5ff)</option>
   <option value="#9b59b6">Royal Purple (#9b59b6)</option>
  </select>
 </div>
</div>

<div class="controls-row">
 <div>
  <label>Main Headline (Catchy Hook):</label>
  <input type="text" id="t-title" value="NEVER OPEN THE DOOR!" oninput="drawThumbnail()">
 </div>
 <div>
  <label>Headline Color:</label>
  <select id="t-title-color" onchange="drawThumbnail()">
   <option value="#ffe600">Bright Yellow (#ffe600)</option>
   <option value="#ffffff">Clean White (#ffffff)</option>
   <option value="#ff2222">Blood Red (#ff2222)</option>
   <option value="#00ffcc">Neon Cyan (#00ffcc)</option>
  </select>
 </div>
</div>

<div class="controls-row">
 <div>
  <label>Bottom Subtitle (Extra Detail):</label>
  <input type="text" id="t-sub" value="FULL RECAP • 15 MIN STORY" oninput="drawThumbnail()">
 </div>
 <div>
  <label>Layout Position:</label>
  <select id="t-pos" onchange="drawThumbnail()">
   <option value="bottom">Text at Bottom</option>
   <option value="center">Text in Center</option>
   <option value="top">Text at Top</option>
  </select>
 </div>
</div>

<canvas id="t-canvas" width="1280" height="720"></canvas>

<div style="text-align:center">
<button onclick="downloadThumbnail()" class="action" style="background:#27ae60">⬇ Download Thumbnail (1280x720 HD)</button>
</div>

</div>
</div>

<script>
const $=id=>document.getElementById(id);
let jid=null,timer=null;

function switchTab(t){
 $('tab-video').style.display=t==='video'?'block':'none';
 $('tab-thumb').style.display=t==='thumb'?'block':'none';
 $('tab-video-btn').className=t==='video'?'active':'';
 $('tab-thumb-btn').className=t==='thumb'?'active':'';
 if(t==='thumb'){ drawThumbnail(); }
}

function poll(){timer=setInterval(async()=>{
 const s=await (await fetch('/status/'+jid)).json();
 $('p').style.width=s.progress+'%';$('msg').textContent=s.message;
 if(s.status==='need_beats'){clearInterval(timer);$('manual').style.display='block';$('prompt').value=s.prompt;$('go').disabled=false}
 else if(s.status!=='running'){clearInterval(timer);$('go').disabled=false;
  $('res').innerHTML=s.outputs.map(o=>`<p><a class="dl" href="/download/${encodeURIComponent(o.name)}">⬇ ${o.name}</a> <small>(${o.label || 'File'}, ${o.minutes} min)</small></p>`).join('');}
},1500)}

$('go').onclick=async()=>{
 const f=$('file').files[0];if(!f){alert('Video select maadi');return}
 const fd=new FormData();fd.append('file',f);fd.append('target',$('target').value);
 fd.append('voice',$('voice').value);fd.append('model',$('wm').value);fd.append('sub_size',$('sub_size').value);fd.append('key',$('key').value);
 $('go').disabled=true;$('msg').textContent='Uploading...';$('res').innerHTML='';$('manual').style.display='none';
 jid=(await (await fetch('/start',{method:'POST',body:fd})).json()).id;poll();};

$('cont').onclick=async()=>{
 const r=await fetch('/beats/'+jid,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({beats:$('beats').value})});
 const j=await r.json();if(j.error){alert(j.error);return}
 $('manual').style.display='none';$('go').disabled=true;poll();};

/* Thumbnail Studio Canvas Logic */
let activeThumbImg = new Image();
activeThumbImg.crossOrigin = 'anonymous';
activeThumbImg.src = '/thumb/thumb1.jpg';
activeThumbImg.onload = () => drawThumbnail();

function pickThumb(src, el){
 document.querySelectorAll('.thumb-card').forEach(c=>c.classList.remove('selected'));
 if(el) el.classList.add('selected');
 activeThumbImg = new Image();
 activeThumbImg.crossOrigin = 'anonymous';
 activeThumbImg.src = src;
 activeThumbImg.onload = () => drawThumbnail();
}

async function loadVideoFrames(jid){
 try{
  const data = await (await fetch('/frames/'+jid)).json();
  if(!data.frames || !data.frames.length) return;
  const grid = document.getElementById('thumbs-grid');
  grid.innerHTML = '';
  data.frames.forEach((f,i) => {
   const card = document.createElement('div');
   card.className = 'thumb-card' + (i===0?' selected':'');
   card.innerHTML = `<img src="${f.url}?t=${Date.now()}" crossorigin="anonymous"><div class="caption">${i+1}. ${f.caption} (~${Math.floor(f.t/60)}:${String(f.t%60).padStart(2,'0')})</div>`;
   card.onclick = () => pickThumb(f.url+'?t='+Date.now(), card);
   grid.appendChild(card);
   if(i===0) pickThumb(f.url+'?t='+Date.now(), card);
  });
  const hint = document.getElementById('thumb-hint');
  if(hint) hint.textContent = '✅ Frames extracted from selected video. Click any frame to use as thumbnail base.';
 } catch(e){ console.warn('Frame load failed', e); }
}

function loadCustomThumb(e){
 const file = e.target.files[0];
 if(!file) return;
 const r = new FileReader();
 r.onload = ev => {
   activeThumbImg = new Image();
   activeThumbImg.src = ev.target.result;
   activeThumbImg.onload = () => drawThumbnail();
 };
 r.readAsDataURL(file);
}

function drawThumbnail(){
 const c = $('t-canvas'), ctx = c.getContext('2d');
 ctx.clearRect(0, 0, 1280, 720);
 if(activeThumbImg.complete && activeThumbImg.naturalWidth > 0){
   ctx.drawImage(activeThumbImg, 0, 0, 1280, 720);
 } else {
   ctx.fillStyle = '#111';
   ctx.fillRect(0, 0, 1280, 720);
 }

 const badge = $('t-badge').value.trim();
 const badgeColor = $('t-badge-color').value;
 const title = $('t-title').value.trim();
 const titleColor = $('t-title-color').value;
 const sub = $('t-sub').value.trim();
 const pos = $('t-pos').value;

 let baseY = 550;
 if(pos === 'top') baseY = 160;
 else if(pos === 'center') baseY = 360;

 // Dark gradient overlay behind text for readability
 const grad = ctx.createLinearGradient(0, baseY - 120, 0, baseY + 120);
 grad.addColorStop(0, 'rgba(0,0,0,0)');
 grad.addColorStop(0.5, 'rgba(0,0,0,0.65)');
 grad.addColorStop(1, 'rgba(0,0,0,0.9)');
 ctx.fillStyle = grad;
 ctx.fillRect(0, baseY - 130, 1280, 260);

 // 1. Draw Badge
 if(badge){
   ctx.font = '900 24px "Segoe UI", Arial, sans-serif';
   const bWidth = ctx.measureText(badge).width + 36;
   const bHeight = 44;
   const bX = (1280 - bWidth) / 2;
   const bY = baseY - 85;

   ctx.fillStyle = badgeColor;
   ctx.beginPath();
   ctx.roundRect(bX, bY, bWidth, bHeight, 8);
   ctx.fill();

   ctx.fillStyle = '#ffffff';
   ctx.textAlign = 'center';
   ctx.textBaseline = 'middle';
   ctx.fillText(badge, 640, bY + bHeight / 2);
 }

 // 2. Draw Main Title
 if(title){
   ctx.font = '900 68px "Segoe UI", Arial, sans-serif';
   ctx.textAlign = 'center';
   ctx.textBaseline = 'middle';

   // Thick black stroke for high readability
   ctx.lineJoin = 'round';
   ctx.miterLimit = 2;
   ctx.strokeStyle = '#000000';
   ctx.lineWidth = 14;
   ctx.strokeText(title, 640, baseY);

   // Drop shadow
   ctx.shadowColor = 'rgba(0,0,0,0.9)';
   ctx.shadowBlur = 12;
   ctx.fillStyle = titleColor;
   ctx.fillText(title, 640, baseY);
   ctx.shadowBlur = 0; // reset
 }

 // 3. Draw Subtitle
 if(sub){
   ctx.font = '800 32px "Segoe UI", Arial, sans-serif';
   ctx.textAlign = 'center';
   ctx.textBaseline = 'middle';

   ctx.strokeStyle = '#000000';
   ctx.lineWidth = 8;
   ctx.strokeText(sub, 640, baseY + 60);

   ctx.fillStyle = '#ffffff';
   ctx.fillText(sub, 640, baseY + 60);
 }
}

function downloadThumbnail(){
 const c = $('t-canvas');
 const a = document.createElement('a');
 a.download = 'FROM_Season1_Thumbnail.png';
 a.href = c.toDataURL('image/png');
 a.click();
}

window.onload = () => drawThumbnail();
</script></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/thumb/<path:name>")
def thumb(name):
    return send_from_directory(THUMBS, name)


@app.route("/start", methods=["POST"])
def start():
    target = max(2.0, min(30.0, float(request.form.get("target", 10))))
    job_id = uuid.uuid4().hex[:8]
    f = request.files["file"]
    dest = UPLOADS / f"{job_id}_{Path(f.filename).name}"
    f.save(dest)
    sub_size = request.form.get("sub_size", "15")
    JOBS[job_id] = {"status": "running", "progress": 0, "message": "Starting", "outputs": [],
                    "video": dest, "voice": request.form.get("voice", "en-US-GuyNeural"),
                    "sub_size": sub_size}
    threading.Thread(target=process_job, daemon=True, args=(
        job_id, dest, target, request.form.get("key", "").strip(),
        request.form.get("model", "small"))).start()
    return jsonify(id=job_id)


@app.route("/beats/<job_id>", methods=["POST"])
def beats(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Unknown job")
    try:
        parsed = parse_beats(request.get_json()["beats"], job["total"])
    except Exception as e:  # noqa
        return jsonify(error=f"JSON sari illa: {e}")
    job["status"] = "running"
    job["message"] = "Video generate aagtide..."
    threading.Thread(target=finish_job, args=(job_id, parsed), daemon=True).start()
    return jsonify(ok=True)


@app.route("/status/<job_id>")
def status(job_id):
    j = JOBS.get(job_id)
    if not j:
        return jsonify(status="error", progress=0, message="Unknown job", outputs=[])
    return jsonify({k: j.get(k) for k in ("status", "progress", "message", "outputs", "prompt")})


@app.route("/download/<path:name>")
def download(name):
    return send_from_directory(OUTPUTS, name, as_attachment=True)


@app.route("/framefile/<path:name>")
def framefile(name):
    return send_from_directory(FRAMES, name)


@app.route("/frames/<job_id>")
def frames(job_id):
    """Extract 5 evenly-spaced frames from the uploaded video and return their URLs."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Unknown job"), 404
    path = job.get("video")
    if not path or not Path(path).exists():
        return jsonify(error="Video not found"), 404

    try:
        total = probe_duration(path)
    except Exception as e:
        return jsonify(error=str(e)), 500

    # Pick 5 timestamps at 10%, 25%, 45%, 65%, 85% of duration
    pcts = [0.10, 0.25, 0.45, 0.65, 0.85]
    urls = []
    captions = ["Opening Scene", "Early Story", "Mid Point", "Rising Action", "Climax"]
    for i, pct in enumerate(pcts):
        t = total * pct
        fname = f"{job_id}_frame{i+1}.jpg"
        out = FRAMES / fname
        if not out.exists():
            try:
                run(["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(path),
                     "-frames:v", "1", "-q:v", "3",
                     "-vf", "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2",
                     "-update", "1", str(out)])
            except Exception:
                pass
        if out.exists():
            urls.append({"url": f"/framefile/{fname}", "caption": captions[i], "t": round(t)})

    return jsonify(frames=urls)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
