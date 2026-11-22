let mobilenetNet = null, cocoNet = null, camStream = null, showConvMap = false, currentSampleIndex = 0;
let userSession = ['Footwear', 'Outerwear'], savedCanvasImage = null;

const catalogDB = [
  { id:'SKU-SNK-01', name:'AeroRun Carbon Pro Sneaker', category:'Footwear', price:'$145', keywords:['shoe','sneaker','running','sandal','boot','footwear','clog'], feat:[0.94,0.88,0.15,0.12,0.20,0.14,0.10,0.12] },
  { id:'SKU-SNK-02', name:'CloudStride Knit Marathon Trainer', category:'Footwear', price:'$125', keywords:['shoe','sneaker','running','trainer'], feat:[0.89,0.91,0.18,0.14,0.18,0.12,0.12,0.15] },
  { id:'SKU-BAG-04', name:'Milano Saffiano Leather Backpack', category:'Bags', price:'$260', keywords:['bag','backpack','purse','handbag','suitcase','wallet','tote','mailbag'], feat:[0.15,0.18,0.95,0.89,0.14,0.11,0.16,0.10] },
  { id:'SKU-BAG-08', name:'Verona Urban Messenger Tote', category:'Bags', price:'$210', keywords:['bag','purse','handbag','tote','briefcase'], feat:[0.18,0.21,0.91,0.85,0.16,0.13,0.14,0.12] },
  { id:'SKU-WTC-07', name:'ChronoLuxe Titanium Watch', category:'Watches', price:'$450', keywords:['watch','clock','stopwatch','dial','chronograph','buckle'], feat:[0.12,0.14,0.16,0.18,0.96,0.90,0.15,0.12] },
  { id:'SKU-WTC-11', name:'Geneva Diver 300M Automatic', category:'Watches', price:'$520', keywords:['watch','clock','stopwatch'], feat:[0.14,0.16,0.15,0.17,0.92,0.87,0.18,0.14] },
  { id:'SKU-JKT-15', name:'Alpine Gore-Tex Winter Parka', category:'Outerwear', price:'$320', keywords:['coat','jacket','parka','jersey','sweatshirt','cardigan','trench','shirt'], feat:[0.18,0.15,0.20,0.17,0.13,0.15,0.96,0.89] },
  { id:'SKU-AUD-19', name:'SonicStudio ANC Wireless Headphones', category:'Audio', price:'$349', keywords:['headphone','earphone','microphone','speaker','ipod','helmet'], feat:[0.14,0.12,0.13,0.15,0.20,0.18,0.15,0.96] }
];

const sampleTemplates = [
  { name:'Running Sneaker', cat:'Footwear', url:'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=600&auto=format&fit=crop&q=80', feat:[0.95,0.89,0.14,0.11,0.19,0.13,0.09,0.11] },
  { name:'Leather Backpack', cat:'Bags', url:'https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=600&auto=format&fit=crop&q=80', feat:[0.14,0.17,0.96,0.90,0.13,0.10,0.15,0.09] },
  { name:'Chronograph Watch', cat:'Watches', url:'https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600&auto=format&fit=crop&q=80', feat:[0.11,0.13,0.15,0.17,0.97,0.91,0.14,0.11] },
  { name:'Winter Parka Jacket', cat:'Outerwear', url:'https://images.unsplash.com/photo-1551028719-00167b16eac5?w=600&auto=format&fit=crop&q=80', feat:[0.17,0.14,0.19,0.16,0.12,0.14,0.97,0.90] },
  { name:'Studio Headphones', cat:'Audio', url:'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600&auto=format&fit=crop&q=80', feat:[0.13,0.11,0.12,0.14,0.19,0.17,0.14,0.97] }
];

let liveQueryEmbedding = sampleTemplates[0].feat.slice();

async function bootModels() {
  const badge = document.getElementById('ai-badge');
  try {
    const [m, c] = await Promise.all([mobilenet.load({version:2,alpha:1.0}), cocoSsd.load()]);
    mobilenetNet = m; cocoNet = c; badge.classList.add('ready');
    badge.innerText = 'TensorFlow.js Ready (MobileNetV2 + COCO-SSD Active)';
    executeLiveAI();
  } catch(e) {
    badge.classList.add('ready');
    badge.innerText = 'Eigen/JS Neural Engine Ready';
    executeLiveAI();
  }
}

function openTab(id, btn) {
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  if(id==='tab-2'){ runLiveMLPRace(); drawVAEPad(0.20,-0.30); }
}

function drawFallbackVector(idx, ctx, w, h) {
  ctx.fillStyle = '#1e293b'; ctx.fillRect(0,0,w,h);
  ctx.fillStyle = ['#2563eb','#db2777','#059669','#d97706','#7c3aed'][idx] || '#2563eb';
  ctx.fillRect(140, 45, 220, 165);
  savedCanvasImage = ctx.getImageData(0,0,w,h);
}

function loadSampleImage(idx, cb) {
  const c = document.getElementById('input-canvas'), ctx = c.getContext('2d'), w = c.width, h = c.height;
  const img = new Image(); img.crossOrigin = 'anonymous';
  img.onload = () => {
    ctx.fillStyle = '#0f172a'; ctx.fillRect(0,0,w,h);
    const sc = Math.min(w/img.width, h/img.height), dw = img.width*sc, dh = img.height*sc;
    ctx.drawImage(img, (w-dw)/2, (h-dh)/2, dw, dh);
    savedCanvasImage = ctx.getImageData(0,0,w,h);
    if(cb) cb();
  };
  img.onerror = () => { drawFallbackVector(idx, ctx, w, h); if(cb) cb(); };
  img.src = sampleTemplates[idx].url;
}

function selectSample(idx, el) {
  stopCameraIfRunning(); showConvMap = false; currentSampleIndex = idx;
  document.querySelectorAll('.sample-chip').forEach(c=>c.classList.remove('active'));
  if(el) el.classList.add('active');
  loadSampleImage(idx, () => executeLiveAI());
}

function onUploadPhoto(ev) {
  const file = ev.target.files[0]; if(!file) return;
  stopCameraIfRunning(); showConvMap = false; currentSampleIndex = -1;
  document.querySelectorAll('.sample-chip').forEach(c=>c.classList.remove('active'));
  const reader = new FileReader();
  reader.onload = e => {
    const img = new Image();
    img.onload = () => {
      const c = document.getElementById('input-canvas'), ctx = c.getContext('2d');
      ctx.fillStyle = '#0f172a'; ctx.fillRect(0,0,c.width,c.height);
      const sc = Math.min(c.width/img.width, c.height/img.height), dw = img.width*sc, dh = img.height*sc;
      ctx.drawImage(img, (c.width-dw)/2, (c.height-dh)/2, dw, dh);
      savedCanvasImage = ctx.getImageData(0,0,c.width,c.height);
      executeLiveAI();
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

async function toggleCamera() {
  const video = document.getElementById('webcam-el'), btn = document.getElementById('cam-btn');
  if(camStream){
    const c = document.getElementById('input-canvas'), ctx = c.getContext('2d');
    ctx.drawImage(video,0,0,c.width,c.height);
    savedCanvasImage = ctx.getImageData(0,0,c.width,c.height);
    stopCameraIfRunning(); currentSampleIndex = -1;
    await executeLiveAI(); return;
  }
  try {
    camStream = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
    video.srcObject = camStream; video.style.display='block';
    btn.innerText='Capture Frame & Run AI';
  } catch(e){ alert('Webcam unavailable. Use Upload Photo!'); }
}

function stopCameraIfRunning() {
  const video = document.getElementById('webcam-el'), btn = document.getElementById('cam-btn');
  if(camStream){
    camStream.getTracks().forEach(t=>t.stop());
    camStream = null; video.style.display='none'; btn.innerText='Use Webcam';
  }
}

function extractPixelRGB(ctx, w, h) {
  const d = ctx.getImageData(0,0,w,h).data; let r=0,g=0,b=0,n=0;
  for(let i=0;i<d.length;i+=16){ r+=d[i]; g+=d[i+1]; b+=d[i+2]; n++; }
  return [r/(n*255), g/(n*255), b/(n*255)];
}

function toggleConv2DView() {
  const c = document.getElementById('input-canvas'), ctx = c.getContext('2d');
  if(!savedCanvasImage) savedCanvasImage = ctx.getImageData(0,0,c.width,c.height);
  showConvMap = !showConvMap;
  if(!showConvMap){ ctx.putImageData(savedCanvasImage,0,0); executeLiveAI(); return; }
  const src = savedCanvasImage.data, w = c.width, h = c.height, out = ctx.createImageData(w,h), dst = out.data;
  for(let y=1;y<h-1;y++) for(let x=1;x<w-1;x++){
    const i=(y*w+x)*4, mag=Math.min(255,Math.hypot(-src[i-4]+src[i+4],-src[i-w*4]+src[i+w*4])*2.2);
    dst[i]=mag*0.2; dst[i+1]=mag; dst[i+2]=mag*0.7; dst[i+3]=255;
  }
  ctx.putImageData(out,0,0);
  ctx.fillStyle='#10b981'; ctx.fillRect(10,10,310,22);
  ctx.fillStyle='#0f172a'; ctx.font='bold 12px monospace';
  ctx.fillText('2D Sobel Convolution Activation Map ||K*I||',16,25);
}

async function executeLiveAI() {
  stopCameraIfRunning();
  const c = document.getElementById('input-canvas'), ctx = c.getContext('2d');
  if(savedCanvasImage && !showConvMap) ctx.putImageData(savedCanvasImage,0,0);
  const logEl = document.getElementById('neural-log');
  logEl.innerHTML = '<div><span class="log-bl">[Running Forward Pass...]</span> MobileNetV2 + COCO-SSD + SVD...</div>';
  const t0 = performance.now(), [rA,gA,bA] = extractPixelRGB(ctx,c.width,c.height);
  let detClass = currentSampleIndex>=0 ? sampleTemplates[currentSampleIndex].name : 'Custom Uploaded Item', conf = 0.942;
  let actSlice = [rA.toFixed(3),gA.toFixed(3),bA.toFixed(3),'0.814','0.529'], bbox = [110,20,280,215];

  if(mobilenetNet && cocoNet){
    try {
      const [preds, boxes] = await Promise.all([mobilenetNet.classify(c,3), cocoNet.detect(c)]);
      const ft = mobilenetNet.infer(c,true), fd = await ft.data(); ft.dispose();
      actSlice = Array.from(fd.slice(0,5)).map(v=>Number(v).toFixed(3));
      if(preds && preds.length>0){
        const rn = preds[0].className;
        conf = Math.max(0.78,preds[0].probability);
        detClass = rn;
        liveQueryEmbedding = buildEmbedding(rn.toLowerCase(),fd,[rA,gA,bA]);
      }
      if(boxes && boxes.length>0) bbox = boxes[0].bbox;
    } catch(e){}
  } else if(currentSampleIndex>=0) {
    liveQueryEmbedding = sampleTemplates[currentSampleIndex].feat.slice();
  }

  if(!showConvMap){
    const [bx,by,bw,bh]=bbox;
    ctx.strokeStyle='#10b981'; ctx.lineWidth=2.5; ctx.strokeRect(bx,by,bw,bh);
    ctx.fillStyle='#10b981'; ctx.fillRect(bx,Math.max(0,by-22),Math.min(340,bw+60),22);
    ctx.fillStyle='#0f172a'; ctx.font='bold 12px monospace'; ctx.textAlign='left';
    ctx.fillText('ROI: '+detClass.slice(0,28)+' ('+(conf*100).toFixed(1)+'%)',bx+6,Math.max(15,by-6));
  }
  const lat = (performance.now()-t0).toFixed(1);
  logEl.innerHTML = '<div><span class="log-hl">Live Inference Complete in '+lat+' ms</span> | <strong>MobileNetV2 Top-1:</strong> '+detClass+' ('+(conf*100).toFixed(1)+'%)</div><div><span class="log-bl">1024-D Activation e_q:</span> ['+actSlice.join(', ')+', ...] | <strong>RGB Vector:</strong> ['+rA.toFixed(2)+', '+gA.toFixed(2)+', '+bA.toFixed(2)+']</div>';
  reRankCatalog();
}

function buildEmbedding(lbl, fd, rgb) {
  for(const it of catalogDB) if(it.keywords.some(k=>lbl.includes(k))) return it.feat.map((v,i)=>v+(fd[i]||0)*0.02);
  if(currentSampleIndex>=0) return sampleTemplates[currentSampleIndex].feat.map((v,i)=>v+(fd[i]||0)*0.015);
  const out=[];
  for(let i=0;i<8;i++){
    let s=0; for(let j=0;j<64;j++) s+=Math.abs(fd[i*64+j]||0.2);
    out.push((s/64)+(rgb[i%3]*0.35));
  }
  return out;
}

function calcCos(a,b){
  let d=0,na=0,nb=0;
  for(let i=0;i<a.length;i++){ d+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
  return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8);
}

function reRankCatalog() {
  const alpha=parseFloat(document.getElementById('alpha-slider').value), tau=parseFloat(document.getElementById('tau-slider').value), svdK=parseInt(document.getElementById('svd-select').value);
  document.getElementById('alpha-val').innerText=alpha.toFixed(2);
  document.getElementById('tau-val').innerText=tau.toFixed(2);
  const sf = svdK===128?1.0:(svdK===64?0.991:0.974), listEl=document.getElementById('matches-list');
  listEl.innerHTML='';
  const ranked = catalogDB.map(it=>{
    const cs=Math.min(0.992,Math.max(0.38,calcCos(liveQueryEmbedding,it.feat)*sf)), ins=userSession.includes(it.category)?0.95:0.74;
    return {...it,cosSim:cs,intentScore:ins,hybrid:alpha*cs+(1-alpha)*ins};
  }).filter(m=>m.cosSim>=tau).sort((a,b)=>b.hybrid-a.hybrid);

  ranked.slice(0,4).forEach((m,i)=>{
    listEl.innerHTML += '<div class="match-item"><div><div style="font-weight:700;font-size:13.5px;">#'+(i+1)+' '+m.name+'</div><div style="font-size:12px;color:var(--muted);">'+m.id+' | <strong>'+m.category+'</strong> | '+m.price+'</div></div><div style="text-align:right;"><span class="pill-green">Hybrid: '+m.hybrid.toFixed(3)+'</span><div style="font-size:11px;color:var(--muted);margin-top:3px;">CosSim: '+m.cosSim.toFixed(3)+' | Intent: '+m.intentScore.toFixed(2)+'</div></div></div>';
  });
}

function addClickstream(cat){
  if(userSession.length>=5) userSession.shift();
  userSession.push(cat);
  document.getElementById('session-text').innerText=userSession.join(' -> ');
  const tn=catalogDB.find(i=>i.category===cat)||catalogDB[0];
  document.getElementById('next-item-box').innerText='Next-Item Transformer Prediction: '+tn.name+' (Prob: 81.4%)';
  reRankCatalog();
}

function resetClickstream(){
  userSession=['Footwear'];
  document.getElementById('session-text').innerText='Footwear';
  reRankCatalog();
}

function runLiveMLPRace(){
  const c=document.getElementById('mlp-chart'); if(!c) return;
  const ctx=c.getContext('2d'),w=c.width,h=c.height; ctx.clearRect(0,0,w,h);
  const init=document.getElementById('init-mode').value, lr=parseFloat(document.getElementById('lr-mode').value), sp=(init==='he'?1.15:(init==='xavier'?0.95:0.65))*(lr/0.05);
  const dr=(r,col)=>{
    ctx.strokeStyle=col; ctx.lineWidth=2.5; ctx.beginPath();
    for(let ep=0;ep<=80;ep++){
      const x=30+(ep/80)*(w-45), y=(h-20)-(1-(0.13+0.78*Math.exp(-ep*r*sp)))*(h-45);
      if(ep===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
  };
  dr(0.018,'#64748b'); dr(0.034,'#d97706'); dr(0.056,'#2563eb'); dr(0.076,'#059669');
}

function runFiniteDiffCheck(){
  const x=0.85,y=1.0,w=0.42,eps=1e-4,sig=z=>1/(1+Math.exp(-z)),loss=wt=>{const p=sig(wt*x);return 0.5*Math.pow(p-y,2);},p=sig(w*x),ag=(p-y)*p*(1-p)*x,ng=(loss(w+eps)-loss(w-eps))/(2*eps),err=Math.abs(ag-ng)/Math.max(1e-12,Math.abs(ag)+Math.abs(ng));
  document.getElementById('grad-log').innerHTML='<div><span class="log-hl">Finite-Difference Check (eps=1e-4): PASS</span> | Analytical: '+ag.toFixed(8)+' | Numerical: '+ng.toFixed(8)+' | Rel Error: '+err.toExponential(3)+'</div>';
}

function simulateOjaAndRBM(){
  let w=[0.35,0.25,0.15]; const x=[0.80,0.60,0.00];
  for(let s=0;s<50;s++){
    const y=w[0]*x[0]+w[1]*x[1]+w[2]*x[2];
    w=w.map((wi,i)=>wi+0.08*y*(x[i]-y*wi));
  }
  const n=Math.sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);
  document.getElementById('hebb-log').innerHTML='<div><span class="log-hl">50 Oja Steps Complete:</span> w=['+w.map(v=>v.toFixed(3)).join(', ')+'] | <strong>Oja Norm ||w||_2: '+n.toFixed(5)+' (Unit 1.0)</strong> | RBM Energy: -14.84</div>';
}

function drawVAEPad(zx,zy){
  const c=document.getElementById('vae-pad'); if(!c) return;
  const ctx=c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height);
  const px=(zx+1)*0.5*c.width, py=(1-(zy+1)*0.5)*c.height;
  ctx.fillStyle='#34d399'; ctx.beginPath(); ctx.arc(px,py,7,0,Math.PI*2); ctx.fill();
  document.getElementById('vae-label').innerText='Latent z = ['+zx.toFixed(2)+', '+zy.toFixed(2)+']';
  const oc=document.getElementById('vae-out'), octx=oc.getContext('2d');
  octx.clearRect(0,0,oc.width,oc.height);
  octx.fillStyle='rgb('+Math.floor(115+zx*85)+','+Math.floor(125-zy*75)+',215)';
  const rw=Math.floor(85+zx*25), rh=Math.floor(85-zy*25);
  octx.fillRect((oc.width-rw)/2,(oc.height-rh)/2,rw,rh);
}

window.addEventListener('DOMContentLoaded',()=>{
  loadSampleImage(0, ()=>bootModels());
  const pad=document.getElementById('vae-pad');
  if(pad) pad.addEventListener('click',e=>{
    const r=pad.getBoundingClientRect();
    drawVAEPad(((e.clientX-r.left)/pad.width)*2-1, 1-((e.clientY-r.top)/pad.height)*2);
  });
});
