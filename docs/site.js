{
document.documentElement.classList.add("js");
const $=s=>document.querySelector(s);
const reduce=matchMedia("(prefers-reduced-motion: reduce)").matches;
const fmt=n=>Math.round(n).toLocaleString("tr-TR");
const dec2=(n,d=2)=>n.toFixed(d).replace(".",",");
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let seed=7;const rnd=()=>(seed=seed*16807%2147483647)/2147483647;
const gauss=()=>{let u=0,v=0;while(!u)u=rnd();while(!v)v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)};
function mkRng(s){let x=(s%2147483646)+1;const r=()=>(x=x*16807%2147483647)/2147483647;return{r,g:()=>{let u=0,v=0;while(!u)u=r();while(!v)v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}}}
function hash(s){let h=2166136261;for(const ch of s){h^=ch.codePointAt(0);h=Math.imul(h,16777619)}return Math.abs(h)}
function onResize(el,fn){const ro=new ResizeObserver(()=>fn());ro.observe(el);return ro}
function bytes(b){const u=["B","KiB","MiB","GiB","TiB"];let i=0;while(b>=1024&&i<4){b/=1024;i++}return(b<10?b.toFixed(2):b<100?b.toFixed(1):b.toFixed(0)).replace(".",",")+" "+u[i]}
const C="#22d3ee",V="#8b5cf6",B="#3b82f6",P="#f472b6";
const visible=new Set();
const vio=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)visible.add(e.target);else visible.delete(e.target)}),{rootMargin:"80px"});
const isOn=el=>visible.has(el);

/* ================= Model hesapları (koklm/analiz.py ile aynı) ================= */
const PRESET={
 kucuk:{V:256,d:256,L:4,H:8,Hkv:2,mo:64,m:"",tied:true,ad:"küçük"},
 "1B":{V:128256,d:2048,L:16,H:32,Hkv:8,mo:256,m:"1.5",tied:true,ad:"Llama 3.2 1B"},
 "8B":{V:128256,d:4096,L:32,H:32,Hkv:8,mo:1024,m:"1.3",tied:false,ad:"Llama 3 8B"}};
function ffn(d,mo,m){let h=4*d;h=Math.floor(2*h/3);if(m)h=Math.floor(parseFloat(m)*h);return mo*Math.ceil(h/mo)}
function hesap(c,T){
 const dh=c.d/c.H,kv=c.Hkv*dh,f=ffn(c.d,c.mo,c.m);
 const att=2*c.d*c.d+2*c.d*kv,mlp=3*c.d*f,norm=2*c.d,blok=att+mlp+norm;
 const emb=c.V*c.d,out=c.tied?0:c.d*c.V,N=emb+c.L*blok+c.d+out;
 return{dh,kv,f,att,mlp,norm,blok,emb,out,N,attT:c.L*att,mlpT:c.L*mlp,normT:c.L*norm+c.d,
  ffnNeuron:c.L*f,allNeuron:c.L*(3*c.d+2*kv+2*f)+c.V,
  kvBytes:2*c.L*c.Hkv*dh*T*2,fwd:2*N+2*c.L*T*c.d}}

/* ================= Global model seçici ================= */
const M={key:"1B",subs:[]};
const cfg=()=>PRESET[M.key];
const onModel=f=>{M.subs.push(f)};
function setModel(k){if(k===M.key)return;M.key=k;document.querySelectorAll("#modelSeg button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.m===k));M.subs.forEach(f=>f())}
$("#modelSeg").addEventListener("click",e=>{const b=e.target.closest("button");if(b)setModel(b.dataset.m)});

/* ================= Canlı nöral ağ arka planı ================= */
{
const cv=$("#net"),ctx=cv.getContext("2d");let W=0,H=0,nodes=[],pulses=[];const mouse={x:-1e4,y:-1e4};
function size(){const dpr=Math.min(2,devicePixelRatio||1);W=innerWidth;H=innerHeight;cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
 const n=Math.round(Math.min(110,W*H/14000));nodes=Array.from({length:n},()=>({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.25,vy:(Math.random()-.5)*.25,r:1+Math.random()*1.8,h:Math.random()}))}
addEventListener("resize",size);size();
addEventListener("pointermove",e=>{mouse.x=e.clientX;mouse.y=e.clientY},{passive:true});
document.addEventListener("pointerleave",()=>{mouse.x=mouse.y=-1e4});
const col=h=>h<.5?`${34+(139-34)*h*2|0},${211-(211-92)*h*2|0},${238+(246-238)*h*2|0}`:`${139+(244-139)*(h-.5)*2|0},${92+(114-92)*(h-.5)*2|0},${246-(246-182)*(h-.5)*2|0}`;
const D=150;
function frame(){ctx.clearRect(0,0,W,H);
 for(const a of nodes){if(!reduce){a.x+=a.vx;a.y+=a.vy;if(a.x<-20)a.x=W+20;if(a.x>W+20)a.x=-20;if(a.y<-20)a.y=H+20;if(a.y>H+20)a.y=-20;
  const dx=mouse.x-a.x,dy=mouse.y-a.y,dd=Math.hypot(dx,dy);if(dd<200&&dd>1){a.x+=dx/dd*.35;a.y+=dy/dd*.35}}}
 ctx.lineWidth=1;
 for(let i=0;i<nodes.length;i++){const a=nodes[i];for(let j=i+1;j<nodes.length;j++){const b=nodes[j],dx=a.x-b.x,dy=a.y-b.y;if(Math.abs(dx)>D||Math.abs(dy)>D)continue;const d=Math.hypot(dx,dy);if(d<D){ctx.strokeStyle=`rgba(${col((a.h+b.h)/2)},${(1-d/D)*.32})`;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
   if(!reduce&&Math.random()<.0007&&pulses.length<26)pulses.push({a,b,t:0})}}
  const md=Math.hypot(mouse.x-a.x,mouse.y-a.y);if(md<180){ctx.strokeStyle=`rgba(244,114,182,${(1-md/180)*.5})`;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(mouse.x,mouse.y);ctx.stroke()}}
 for(const a of nodes){ctx.fillStyle=`rgba(${col(a.h)},.9)`;ctx.beginPath();ctx.arc(a.x,a.y,a.r,0,7);ctx.fill()}
 ctx.globalCompositeOperation="lighter";
 pulses=pulses.filter(p=>{p.t+=.018;if(p.t>1)return false;const x=p.a.x+(p.b.x-p.a.x)*p.t,y=p.a.y+(p.b.y-p.a.y)*p.t;const g=ctx.createRadialGradient(x,y,0,x,y,9);g.addColorStop(0,"rgba(255,255,255,.95)");g.addColorStop(.3,`rgba(${col(p.a.h)},.7)`);g.addColorStop(1,"rgba(0,0,0,0)");ctx.fillStyle=g;ctx.beginPath();ctx.arc(x,y,9,0,7);ctx.fill();return true});
 ctx.globalCompositeOperation="source-over";
 if(!reduce)requestAnimationFrame(frame)}
requestAnimationFrame(frame);
}

/* ================= Blok seçimi (3B ↔ anatomi) ================= */
const SEL={blk:0,subs:[]};
function selectBlock(i,scroll){SEL.blk=i;SEL.subs.forEach(f=>f());if(scroll)document.querySelector("#blok").scrollIntoView({behavior:reduce?"auto":"smooth"})}

/* ================= Kahraman: 3B transformer yığını ================= */
{
const st=$("#stage"),cv=$("#stageC"),ctx=cv.getContext("2d");let W=0,H=0,yaw=.6,pitch=-.42,drag=null,tm=0,hover=-1,lastHud=0;const vy=reduce?0:.0035;
vio.observe(st);
const X0=-2,X1=2,Z=1.15,split=X0+(X1-X0)*.172,STACK=4.32,y0=-2.05;
const FACES=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[3,2,6,7],[0,3,7,4],[1,2,6,5]];
function box(x0,x1,y,h,z){return[[x0,y,-z],[x1,y,-z],[x1,y+h,-z],[x0,y+h,-z],[x0,y,z],[x1,y,z],[x1,y+h,z],[x0,y+h,z]]}
let L=16,gap=.27,slabs=[];
function build(){L=cfg().L;gap=STACK/L;const sh=gap*.56;const fr=hesap(cfg(),2048);const sp=X0+(X1-X0)*fr.att/(fr.att+fr.mlp);slabs=[];
 for(let i=0;i<L;i++){const y=y0+i*gap;slabs.push({i,v:box(X0,sp-.03,y,sh,Z),c:[139,92,246]},{i,v:box(sp+.03,X1,y,sh,Z),c:[34,211,238]})}}
build();
const emb=[];for(let a=0;a<10;a++)for(let b=0;b<4;b++)emb.push([X0+.2+a*.4,y0-.75,-Z+.29+b*.57]);
const ringY=()=>y0+STACK+.55;
const parts=Array.from({length:70},(_,i)=>({t:Math.random(),e:i%40,s:.0018+Math.random()*.0016,ph:Math.random()*6.28,o:(i%2?1:-1)*(.3+Math.random()*1.3)}));
function size(){const dpr=Math.min(2,devicePixelRatio||1);W=st.clientWidth;H=st.clientHeight;cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw()}
function proj(p){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const x=p[0]*cy-p[2]*sy,z=p[0]*sy+p[2]*cy,y=p[1];const y2=y*cp-z*sp,z2=y*sp+z*cp;const D=9.5,f=Math.min(W*1.05,H*1.25);const k=f/(D+z2);return[W/2+x*k,H*.53-y2*k,z2,k]}
function glow(x,y,r,c,a){const g=ctx.createRadialGradient(x,y,0,x,y,r);g.addColorStop(0,`rgba(255,255,255,${a})`);g.addColorStop(.25,`rgba(${c},${a*.8})`);g.addColorStop(1,`rgba(${c},0)`);ctx.fillStyle=g;ctx.beginPath();ctx.arc(x,y,r,0,7);ctx.fill()}
let faces=[],wave=0;
function draw(){if(!W)return;ctx.clearRect(0,0,W,H);
 wave=(tm*.00045)%1.25*L;
 const base=proj([0,y0-.8,0]);const g0=ctx.createRadialGradient(base[0],base[1],0,base[0],base[1],W*.45);g0.addColorStop(0,"rgba(139,92,246,.22)");g0.addColorStop(1,"rgba(139,92,246,0)");ctx.fillStyle=g0;ctx.fillRect(0,0,W,H);
 // zemin ızgarası
 ctx.strokeStyle="rgba(139,92,246,.18)";ctx.lineWidth=1;
 for(let k=-4;k<=4;k++){let a=proj([k*.6,y0-.95,-2.4]),b=proj([k*.6,y0-.95,2.4]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();a=proj([-2.4,y0-.95,k*.6]);b=proj([2.4,y0-.95,k*.6]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()}
 ctx.globalCompositeOperation="lighter";
 emb.forEach((p,i)=>{const q=proj(p);glow(q[0],q[1],q[3]*.13,i%3?"34,211,238":"139,92,246",.55+.35*Math.sin(tm*.003+i))});
 ctx.globalCompositeOperation="source-over";
 faces=[];slabs.forEach(s=>{const Pp=s.v.map(proj);let lit=Math.max(0,1-Math.abs(s.i-wave)/1.6);if(s.i===hover)lit=1.2;if(s.i===SEL.blk)lit=Math.max(lit,.75);FACES.forEach((f,fi)=>{const z=f.reduce((a,k)=>a+Pp[k][2],0)/4;faces.push({P:Pp,f,z,c:s.c,lit,fi,i:s.i})})});
 faces.sort((a,b)=>b.z-a.z);
 for(const F of faces){ctx.beginPath();F.f.forEach((k,j)=>{const p=F.P[k];j?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])});ctx.closePath();
  const shade=F.fi===3?1.25:F.fi===2?.55:.85;const a=Math.min(.85,(.13+F.lit*.42)*shade);
  ctx.fillStyle=`rgba(${F.c[0]},${F.c[1]},${F.c[2]},${a})`;ctx.fill();
  ctx.strokeStyle=F.i===hover||F.i===SEL.blk?"rgba(255,255,255,.95)":`rgba(${Math.min(255,F.c[0]+80)},${Math.min(255,F.c[1]+60)},255,${.35+F.lit*.6})`;ctx.lineWidth=1+Math.min(1,F.lit)*1.2;ctx.stroke()}
 ctx.globalCompositeOperation="lighter";
 for(let k=0;k<48;k++){const t=k/48*Math.PI*2,q=proj([Math.cos(t)*1.8,ringY(),Math.sin(t)*1.1]);glow(q[0],q[1],q[3]*.09,"244,114,182",.45+.4*Math.sin(tm*.004+k*.5))}
 for(const p of parts){const e=emb[p.e];let pt;
  if(p.t<.15){const u=p.t/.15;pt=[e[0]*(1-u*.7),e[1]+u*.7,e[2]*(1-u*.7)]}
  else if(p.t<.85){const u=(p.t-.15)/.7,y=y0-.05+u*(STACK+.1),ang=p.ph+u*5;pt=[Math.cos(ang)*p.o,y,Math.sin(ang)*Math.min(.9,Math.abs(p.o))*.8]}
  else{const u=(p.t-.85)/.15,t=((p.e*7)%48)/48*Math.PI*2,r=[Math.cos(t)*1.8,ringY(),Math.sin(t)*1.1],s=[Math.cos(p.ph+5)*p.o,ringY()-.5,0];pt=[s[0]+(r[0]-s[0])*u,s[1]+(r[1]-s[1])*u,s[2]+(r[2]-s[2])*u]}
  const q=proj(pt);glow(q[0],q[1],q[3]*.14,p.t>.85?"244,114,182":"165,243,252",.95)}
 ctx.globalCompositeOperation="source-over";
 if(W>380){ctx.font="600 13px Inter,sans-serif";ctx.textAlign="left";
  let rx=0;for(const F of faces)for(const k of F.f)rx=Math.max(rx,F.P[k][0]);rx=Math.min(rx+18,W-200);
  const lab=(y,t,s,c)=>{const q=proj([0,y,0]);ctx.strokeStyle="rgba(255,255,255,.25)";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(rx-6,q[1]-4);ctx.lineTo(rx-14,q[1]-4);ctx.stroke();ctx.fillStyle=c;ctx.fillText(t,rx,q[1]);if(s){ctx.fillStyle="#8d95b5";ctx.font="12px Inter,sans-serif";ctx.fillText(s,rx,q[1]+16);ctx.font="600 13px Inter,sans-serif"}};
  const c=cfg();
  lab(ringY(),"Çıkış",`${fmt(c.V)} olasılık`,"#f9a8d4");
  lab(y0+STACK*.55,`${L} transformer bloğu`,"mor: attention · mavi: FFN","#e8ebf7");
  lab(y0-.8,"Embedding",`token → ${fmt(c.d)} boyut`,"#a5f3fc")}
 if(hover>=0){const s=slabs[hover*2+1],q=proj([X1,s.v[2][1],0]);ctx.font="600 13px Inter,sans-serif";const txt=`blok ${hover+1} · tıkla`;const w=ctx.measureText(txt).width+16;ctx.fillStyle="rgba(10,13,31,.85)";ctx.strokeStyle=C;ctx.beginPath();if(ctx.roundRect)ctx.roundRect(Math.min(q[0]+10,W-w-4),q[1]-22,w,26,8);else ctx.rect(Math.min(q[0]+10,W-w-4),q[1]-22,w,26);ctx.fill();ctx.stroke();ctx.fillStyle="#fff";ctx.fillText(txt,Math.min(q[0]+18,W-w+4),q[1]-4)}
}
function hud(){const c=cfg(),r=hesap(c,2048);$("#hudTL").innerHTML=`MODEL <b>${M.key}</b><br>d=${c.d} · L=${c.L}<br>H=${c.H} · Hkv=${c.Hkv} · dh=${r.dh}`;
 $("#hudTR").innerHTML=`aktif blok <b>${String(Math.min(L,Math.floor(wave)+1)).padStart(2,"0")}/${L}</b><br>ışık ${parts.length}<br>seçili <b>${SEL.blk+1}</b>`;
 $("#hudBL").innerHTML=`yaw ${Math.round(((yaw*180/Math.PI)%360+360)%360)}° · pitch ${Math.round(pitch*180/Math.PI)}°`}
function loop(t){if(isOn(st)||!tm){tm=t;if(!drag)yaw+=vy;for(const p of parts)p.t=(p.t+p.s)%1;draw();if(t-lastHud>120){hud();lastHud=t}}requestAnimationFrame(loop)}
function inPoly(x,y,pts){let ins=false;for(let i=0,j=pts.length-1;i<pts.length;j=i++){const a=pts[i],b=pts[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])ins=!ins}return ins}
function pick(e){const r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best=-1,bz=1e9;for(const F of faces){if(inPoly(x,y,F.f.map(k=>F.P[k]))&&F.z<bz){bz=F.z;best=F.i}}return best}
st.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY,yaw,pitch,moved:false};st.setPointerCapture(e.pointerId)});
st.addEventListener("pointermove",e=>{if(!drag){const h=pick(e);if(h!==hover){hover=h;st.style.cursor=h>=0?"pointer":"grab";if(reduce)draw()}return}
 if(Math.abs(e.clientX-drag.x)+Math.abs(e.clientY-drag.y)>5)drag.moved=true;yaw=drag.yaw+(e.clientX-drag.x)*.008;pitch=Math.max(-1.1,Math.min(.25,drag.pitch+(e.clientY-drag.y)*.005));if(reduce)draw()});
st.addEventListener("pointerup",e=>{const d=drag;drag=null;if(d&&!d.moved){const i=pick(e);if(i>=0)selectBlock(i,true)}});
st.addEventListener("pointercancel",()=>{drag=null});
st.addEventListener("pointerleave",()=>{hover=-1});
onResize(st,size);if(reduce){tm=1;draw();hud()}else requestAnimationFrame(loop);
onModel(()=>{build();if(SEL.blk>=L)SEL.blk=0;draw();hud()});
SEL.subs.push(()=>{draw();hud()});
// kahraman sayıları
const heroNums=(anim)=>{const c=cfg(),r=hesap(c,2048),el=$("#heroCount");$("#heroCountL").textContent=`parametre · ${c.ad} ile birebir aynı mimari`;$("#kL").textContent=c.L;$("#kN").textContent=fmt(r.ffnNeuron);
 if(reduce||!anim){el.textContent=fmt(r.N);return}const from=parseInt(el.textContent.replace(/\./g,""))||0,t0=performance.now();const tick=t=>{const k=Math.min(1,(t-t0)/1800),e=1-Math.pow(1-k,4);el.textContent=fmt(from+(r.N-from)*e);if(k<1)requestAnimationFrame(tick)};requestAnimationFrame(tick)};
$("#heroCount").textContent="0";heroNums(true);onModel(()=>heroNums(true));
}

/* ================= Python sözdizimi renklendirme ================= */
const KW=new Set("def class return if else elif for in while not and or is None True False import from as with try except finally raise lambda yield pass break continue assert global nonlocal".split(" "));
function hiPy(code,hl){let inDoc=false;return code.split("\n").map((line,i)=>{let out="";
 if(inDoc){const k=line.indexOf('"""');if(k<0)return{h:`<span class="s">${esc(line)}</span>`,i};out+=`<span class="s">${esc(line.slice(0,k+3))}</span>`;line=line.slice(k+3);inDoc=false}
 const re=/(#.*$)|("""|''')|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)|\b([A-Za-z_]\w*)\b/g;let last=0,m;
 while((m=re.exec(line))){out+=esc(line.slice(last,m.index));last=re.lastIndex;
  if(m[1])out+=`<span class="cm">${esc(m[1])}</span>`;
  else if(m[2]){const rest=line.slice(last),k=rest.indexOf(m[2]);if(k>=0){out+=`<span class="s">${esc(m[2]+rest.slice(0,k+3))}</span>`;last+=k+3;re.lastIndex=last}else{out+=`<span class="s">${esc(line.slice(m.index))}</span>`;last=line.length;inDoc=true;break}}
  else if(m[3])out+=`<span class="s">${esc(m[3])}</span>`;
  else if(m[4])out+=`<span class="nu">${m[4]}</span>`;
  else if(m[5]){const w=m[5];if(w==="self")out+=`<span class="sf">self</span>`;else if(KW.has(w))out+=`<span class="k">${w}</span>`;else if(line[last]==="(")out+=`<span class="fn">${w}</span>`;else out+=esc(w)}}
 out+=esc(line.slice(last));return{h:out,i}}).map(({h,i},j,arr)=>h)}
function codeHTML(k,hl){const lines=hiPy(k.kod);const raw=k.kod.split("\n");return lines.map((h,i)=>`<span class="l${hl&&hl.test(raw[i])?" hl":""}" data-n="${k.satir+i}">${h||" "}</span>`).join("")}
const KODM=Object.fromEntries((window.KOD||[]).map(k=>[k.id,k]));

/* ================= Veri akışı ================= */
{
const ST=[
 {k:"T",ad:"Token",kod:"bpe",hl:/merges|_birlestir|sayac\[/,eq:"metin → ön-parça → BPE → [id₁, id₂, …]",
  adim:["Metin ön-parçalara bölünür: kelime + önündeki boşluk, sayı, noktalama.","Her parça UTF-8 baytlarına çevrilir; Türkçe harfler 2 bayt tutar.","Öğrenilmiş birleştirmeler sırayla uygulanır, bayt grupları tek bir id olur.","Çıkış: (B, T) boyutunda tam sayı tensörü."],
  ornek:()=>{const s="Bir varmış",b=[...new TextEncoder().encode(s)];return`<p>"${s}" ${s.length} karakter ama <b>${b.length} bayt</b>, çünkü ı ve ş ikişer bayt:</p><div class="eq">${b.map(x=>x.toString(16).padStart(2,"0")).join(" ")}</div><p>Byte tokenizer'ı bunu ${b.length} token yapar. BPE, sık geçen çiftleri birleştirerek aynı metni çok daha az token'a indirir; bu yüzden aynı bağlam penceresine daha çok metin sığar.</p>`}},
 {k:"E",ad:"Embedding",kod:"ileri",hl:/tok_embeddings/,eq:"h₀ = E[x],   E ∈ ℝ^(V×d)",
  adim:["Her id, tablonun bir satırını seçer (one-hot × E ile aynı şey).","Satır, token'ın öğrenilmiş d boyutlu vektörüdür.","Tied ayarda aynı tablo çıkışta da kullanılır: W_out = Eᵀ."],
  ornek:(c,r)=>`<div class="eq">V × d = ${fmt(c.V)} × ${fmt(c.d)} = ${fmt(r.emb)}</div><p>Bu tablo tek başına <b>${dec2(100*r.emb/r.N,1)}%</b> parametre ve bf16'da <b>${bytes(r.emb*2)}</b>. ${c.tied?"1B'de çıkış katmanı bu tabloyu paylaşır, ayrı bir 262 milyonluk matris yok.":"8B'de çıkış katmanı ayrı bir matristir; bu yüzden embedding iki kez sayılır."}</p>`},
 {k:"N",ad:"RMSNorm",kod:"rms",hl:/rsqrt|pow/,eq:"x / √(mean(x²)+ε) ⊙ g",
  adim:["x²'lerin ortalaması alınır (fp32'de, bf16 hassasiyeti yetmez).","Karekökün tersiyle çarpılır: vektörün RMS'i 1 olur.","Öğrenilen g vektörüyle eleman eleman ölçeklenir."],
  ornek:()=>`<p>d = 4 boyutlu küçük bir örnek, g = 1:</p><div class="eq">x        = [ 3, −4, 0, 0 ]
mean(x²) = (9 + 16 + 0 + 0) / 4 = 6,25
rms      = √6,25 = 2,5
x / rms  = [ 1,2, −1,6, 0, 0 ]</div><p>Yön aynı, büyüklük sabit. LayerNorm'dan farkı: ortalama çıkarılmaz, bias yoktur.</p>`},
 {k:"A",ad:"GQA + RoPE",kod:"att",hl:/rope_uygula|repeat_interleave|enable_gqa|scaled_dot_product/,eq:"softmax(QKᵀ/√dh + M)·V",
  adim:["Q, K, V projeksiyonları: x·W_q, x·W_k, x·W_v.","RoPE, Q ve K'yı konuma göre döndürür.","K/V kafaları gruplara paylaştırılır (GQA).","Skorlar √dh'ye bölünür, maske eklenir, softmax alınır.","Ağırlıklar V ile çarpılır, kafalar birleşir, W_o ile d'ye dönülür."],
  ornek:(c,r)=>`<p>B = 1, T = 8 için ${M.key} tensör şekilleri:</p><div class="eq">Q    (1, 8, ${c.H}, ${r.dh})
K, V (1, 8, ${c.Hkv}, ${r.dh})   ← ${c.H/c.Hkv} Q kafası bir K/V'yi paylaşır
skor (1, ${c.H}, 8, 8)
çıkış (1, 8, ${fmt(c.d)})</div><p>Blok başına ${fmt(r.att)} parametre. Token başına hesap: projeksiyonlar ${dec2(2*r.att/1e6,1)} MFLOP + skorlar 4·T·d (T = 2.048'de ${dec2(4*2048*c.d/1e6,1)} MFLOP).</p>`},
 {k:"+",ad:"Residual",kod:"blok",hl:/x \+|h \+/,eq:"h' = h + Attn(RMSNorm(h))",
  adim:["Alt katmanın çıktısı girdiye eklenir.","Normalizasyon sadece alt katmanın girişinde (pre-norm); ana yol normalize edilmez.","Geri yayılımda gradyan bu toplamdan doğrudan akar."],
  ornek:()=>`<div class="eq">∂h'/∂h = I + ∂Attn/∂h</div><p>Türevdeki <b>I</b> sayesinde gradyan, alt katman ne kadar küçük türev üretirse üretsin kaybolmadan geçer. 16 ya da 32 blok derinliğinde eğitimin mümkün olmasının sebebi bu kısa yol.</p>`},
 {k:"F",ad:"SwiGLU FFN",kod:"ffn",hl:/silu|w1|w3|w2/,eq:"(SiLU(xW₁) ⊙ xW₃)·W₂",
  adim:["x iki ayrı matrisle f boyutuna açılır: kapı (W₁) ve değer (W₃).","Kapı SiLU'dan geçer, değerle eleman eleman çarpılır.","W₂ ile tekrar d boyutuna inilir."],
  ornek:(c,r)=>`<div class="eq">(T, ${fmt(c.d)}) → W₁, W₃ → (T, ${fmt(r.f)}) × 2
            → SiLU ⊙ → (T, ${fmt(r.f)})
            → W₂ → (T, ${fmt(c.d)})</div><p>Blok başına ${fmt(r.mlp)} parametre, bloğun <b>${dec2(100*r.mlp/r.blok,1)}%</b>'i. Token başına ${dec2(2*r.mlp/1e6,0)} MFLOP: modelin hesabının çoğu burada.</p>`},
 {k:"×",ad:"Tekrar",kod:"ileri",hl:/for layer|checkpoint/,eq:"h_{l+1} = h' + FFN(RMSNorm(h'))",
  adim:["Aynı blok L kez üst üste dizilir.","Her bloğun kendi ağırlıkları vardır.","Eğitimde gradient checkpointing ile ara aktivasyonlar saklanmayabilir."],
  ornek:(c,r)=>`<div class="eq">${c.L} × ${fmt(r.blok)} = ${fmt(c.L*r.blok)}</div><p>Bloklar toplam parametrenin <b>${dec2(100*c.L*r.blok/r.N,1)}%</b>'i. İleri geçiş token başına ≈ 2N = ${dec2(2*r.N/1e9,2)} GFLOP.</p>`},
 {k:"O",ad:"Çıkış",kod:"ileri",hl:/self\.output|sadece_son|cross_entropy/,eq:"z = RMSNorm(h)·W_out",
  adim:["Son gizli durum normalize edilir.","Sözlükteki her token için bir puan (logit) üretilir.","Eğitimde çapraz entropi, üretimde sadece son konumun logit'i."],
  ornek:(c)=>`<p>Her konum için ${fmt(c.V)} logit. fp32'de token başına <b>${bytes(c.V*4)}</b>; T = 2.048'lik bir dizi için <b>${bytes(c.V*4*2048)}</b>. Üretimde <code>sadece_son=True</code> ile yalnızca son konum hesaplanır, bu bellek tasarrufu oradan gelir.</p>`},
 {k:"p",ad:"Örnekleme",kod:"uret",hl:/ornekle|cache_kur|start_pos=pos/,eq:"p = softmax(z/τ), top-p",
  adim:["Logit'ler sıcaklık τ'ya bölünür: küçük τ keskin, büyük τ yaratıcı.","Softmax ile olasılığa çevrilir.","Top-p: kümülatif olasılığı p'yi geçene kadar en olası token'lar tutulur.","Seçilen token dizinin sonuna eklenir; KV-cache ile sadece o token işlenir."],
  ornek:()=>{const z=[2,1,.1];const sm=(a)=>{const m=Math.max(...a),e=a.map(v=>Math.exp(v-m)),s=e.reduce((x,y)=>x+y);return e.map(v=>v/s)};const p1=sm(z),p2=sm(z.map(v=>v/.5));return`<div class="eq">z           = [ 2,0  1,0  0,1 ]
τ = 1,0 →  p = [ ${p1.map(v=>dec2(v,3)).join("  ")} ]
τ = 0,5 →  p = [ ${p2.map(v=>dec2(v,3)).join("  ")} ]</div><p>τ = 1 ve top-p = 0,9'da ilk iki token tutulur (${dec2(p1[0],3)} + ${dec2(p1[1],3)} = ${dec2(p1[0]+p1[1],3)} ≥ 0,9), üçüncüsü elenir, kalanlar yeniden normalize edilir.</p>`}}];
const fig=$("#flowfig"),svg=$("#flowSvg"),det=$("#detail");vio.observe(fig);
let pts=[],cur=0,auto=!reduce,t0=performance.now(),pathEl=null,lens=[],dot=null,halo=null,tab="denklem";
function layout(){const W=fig.clientWidth,H=fig.clientHeight,n=ST.length;if(!W)return;const vert=W<700;
 pts=ST.map((s,i)=>vert?[W*(i%2?.66:.34),40+i*(H-80)/(n-1)]:[60+i*(W-120)/(n-1),H*.42+Math.sin(i*1.15)*H*.2]);
 let d=`M${pts[0][0]},${pts[0][1]}`;for(let i=1;i<n;i++){const a=pts[i-1],b=pts[i];d+=vert?` C${a[0]},${(a[1]+b[1])/2} ${b[0]},${(a[1]+b[1])/2} ${b[0]},${b[1]}`:` C${(a[0]+b[0])/2},${a[1]} ${(a[0]+b[0])/2},${b[1]} ${b[0]},${b[1]}`}
 let s=`<defs><linearGradient id="fg" x1="0" x2="${vert?0:1}" y1="0" y2="${vert?1:0}"><stop offset="0" stop-color="${C}"/><stop offset=".5" stop-color="${V}"/><stop offset="1" stop-color="${P}"/></linearGradient><filter id="gl" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
 s+=`<path d="${d}" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="10" stroke-linecap="round"/><path id="fp" d="${d}" fill="none" stroke="url(#fg)" stroke-width="3" stroke-linecap="round" stroke-dasharray="2 12" opacity=".9">${reduce?"":`<animate attributeName="stroke-dashoffset" from="0" to="-28" dur="1s" repeatCount="indefinite"/>`}</path>`;
 ST.forEach((st,i)=>{const[x,y]=pts[i];const lx=vert?(i%2?x+34:x-34):x,ly=vert?y+5:y+(Math.sin(i*1.15)>0?-36:46),an=vert?(i%2?"start":"end"):"middle";
  s+=`<g class="node" data-i="${i}" tabindex="0" role="button" aria-label="${st.ad}"><circle class="bg" cx="${x}" cy="${y}" r="24" fill="#0a0d1f" stroke="url(#fg)" stroke-width="2"/><circle class="ring" cx="${x}" cy="${y}" r="30" fill="none" stroke="${C}" stroke-width="2" opacity="0" filter="url(#gl)"/><text x="${x}" y="${y+6}" text-anchor="middle" font-size="18" font-weight="700" fill="#e8ebf7">${st.k}</text><text x="${lx}" y="${ly}" text-anchor="${an}" font-size="14" font-weight="600" fill="#cfd5ec">${st.ad}</text></g>`});
 s+=`<circle id="fh" r="22" fill="${C}" opacity=".3" filter="url(#gl)"/><circle id="fd" r="7" fill="#fff" filter="url(#gl)"/>`;
 svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.innerHTML=s;pathEl=svg.querySelector("#fp");dot=svg.querySelector("#fd");halo=svg.querySelector("#fh");
 const total=pathEl.getTotalLength();lens=pts.map(p=>{let best=0,bd=1e9;for(let l=0;l<=total;l+=total/400){const q=pathEl.getPointAtLength(l),dd=(q.x-p[0])**2+(q.y-p[1])**2;if(dd<bd){bd=dd;best=l}}return best});
 mark();move(0)}
function mark(){svg.querySelectorAll(".node").forEach((g,j)=>{const on=j===cur;g.querySelector(".ring").setAttribute("opacity",on?1:0);g.querySelector(".bg").setAttribute("fill",on?"#2e1065":"#0a0d1f")})}
function move(k){if(!pathEl||!lens.length)return;const a=lens[cur],b=lens[(cur+1)%ST.length];const l=cur===ST.length-1?a:a+(b-a)*k;const q=pathEl.getPointAtLength(l);dot.setAttribute("cx",q.x);dot.setAttribute("cy",q.y);halo.setAttribute("cx",q.x);halo.setAttribute("cy",q.y)}
function body(s,c,r){if(tab==="ornek")return s.ornek(c,r);if(tab==="kod"){const k=KODM[s.kod];return k?`<p class="small"><code>${esc(k.dosya)}</code>, satır ${k.satir}. Vurgulu satırlar bu durağa ait.</p><pre class="code">${codeHTML(k,s.hl)}</pre>`:""}
 return`<div class="eq">${esc(s.eq)}</div><ol class="steps">${s.adim.map(a=>`<li>${esc(a)}</li>`).join("")}</ol>`}
function show(i){cur=i;const s=ST[i],c=cfg(),r=hesap(c,2048);mark();move(0);
 const pv={E:r.emb,N:c.d,A:r.att,F:r.mlp,"×":c.L*r.blok,O:c.tied?c.d:c.d+r.out}[s.k]||0;
 det.innerHTML=`<div><h3 class="gt">${s.ad}</h3><div class="tabs" role="tablist">${[["denklem","Denklem"],["ornek",`Örnek (${M.key})`],["kod","Kod"]].map(([k,t])=>`<button role="tab" data-t="${k}" aria-selected="${k===tab}">${t}</button>`).join("")}</div><div>${body(s,c,r)}</div><div class="prog" ${auto?"":"hidden"}><i id="pg"></i></div></div><div class="facts"><div class="fact"><b>${pv?fmt(pv):"0"}</b><span>parametre (${M.key}${["A","F","N"].includes(s.k)?", blok başına":""})</span></div><div class="fact" style="border-color:${C}"><b>${pv?dec2(100*(["A","F","N"].includes(s.k)?pv*c.L:pv)/r.N,1)+"%":"—"}</b><span>modelin payı</span></div><div class="fact" style="border-color:${P}"><b style="font-size:20px">${esc(c.ad)}</b><span>d = ${fmt(c.d)} · L = ${c.L} · H = ${c.H} · Hkv = ${c.Hkv}</span></div></div>`;t0=performance.now()}
svg.addEventListener("click",e=>{const g=e.target.closest(".node");if(!g)return;auto=false;show(+g.dataset.i)});
svg.addEventListener("keydown",e=>{const g=e.target.closest(".node");if(g&&(e.key==="Enter"||e.key===" ")){e.preventDefault();auto=false;show(+g.dataset.i)}});
det.addEventListener("click",e=>{const b=e.target.closest(".tabs button");if(!b)return;auto=false;tab=b.dataset.t;show(cur)});
onResize(fig,layout);show(0);onModel(()=>show(cur));
function tick(t){if(auto&&isOn(fig)){const k=(t-t0)/4200;const pg=$("#pg");if(pg)pg.style.width=Math.min(100,k*100)+"%";move(Math.max(0,(k-.6)/.4));if(k>=1)show((cur+1)%ST.length)}else if(auto)t0=t-((t-t0)%4200);if(auto)requestAnimationFrame(tick)}
if(auto)requestAnimationFrame(tick);
}

/* ================= Bir bloğun içi (anatomi) ================= */
{
const fig=$("#anatfig"),svg=$("#anatSvg");vio.observe(fig);let sel="w1",path=[],dots=[],rpath=[];
function items(){const c=cfg(),r=hesap(c,2048),d=c.d,kv=r.kv,f=r.f;return{
 n1:{ad:"RMSNorm (attention öncesi)",sh:`g ∈ ℝ^${d}`,p:d,fl:0,rol:"Girişi ölçekler; ana residual yol normalize edilmez."},
 wq:{ad:"W_q (sorgu)",sh:`${d} × ${d}`,p:d*d,fl:2*d*d,rol:`Her token için ${c.H} kafalık sorgu üretir; RoPE bunun üstüne uygulanır.`},
 wk:{ad:"W_k (anahtar)",sh:`${d} × ${kv}`,p:d*kv,fl:2*d*kv,rol:`Sadece ${c.Hkv} K kafası (GQA). Q'dan ${c.H/c.Hkv} kat dar; KV-cache'e bu girer.`},
 wv:{ad:"W_v (değer)",sh:`${d} × ${kv}`,p:d*kv,fl:2*d*kv,rol:"Attention ağırlıklarının toplayacağı içerik. W_k ile aynı genişlikte."},
 at:{ad:"Attention çekirdeği",sh:`(${c.H}, T, T)`,p:0,fl:4*2048*d,rol:"QKᵀ/√dh + maske → softmax → ·V. Parametresi yok; hesabı T ile büyür (burada T = 2.048)."},
 wo:{ad:"W_o (çıkış projeksiyonu)",sh:`${d} × ${d}`,p:d*d,fl:2*d*d,rol:"Birleşmiş kafaları residual'a yazar. Başlangıçta 0,02/√2L ile küçültülür."},
 n2:{ad:"RMSNorm (FFN öncesi)",sh:`g ∈ ℝ^${d}`,p:d,fl:0,rol:"FFN girişini ölçekler."},
 w1:{ad:"W₁ (kapı)",sh:`${d} × ${f}`,p:d*f,fl:2*d*f,rol:"SiLU'dan geçecek kapı sinyali. Bloktaki en büyük üç matristen biri."},
 w3:{ad:"W₃ (yukarı)",sh:`${d} × ${f}`,p:d*f,fl:2*d*f,rol:"Kapıyla çarpılacak değer. SwiGLU'yu klasik FFN'den ayıran üçüncü matris."},
 sg:{ad:"SiLU ⊙",sh:`(T, ${f})`,p:0,fl:3*f,rol:`${fmt(f)} nöron: kapı SiLU'dan geçer, değerle çarpılır. "FFN nöronu" dediğimiz birimler bunlar.`},
 w2:{ad:"W₂ (aşağı)",sh:`${f} × ${d}`,p:f*d,fl:2*f*d,rol:"f boyutundan tekrar d'ye iner ve residual'a yazar."}}}
function draw(){const W=fig.clientWidth,H=fig.clientHeight;if(!W)return;const c=cfg(),r=hesap(c,2048),I=items();
 const cx=W*.56,maxW=W*.74,u=maxW/(2*r.f),bh=Math.max(30,H*.05),minw=W<500?54:70;
 const wOf=dim=>Math.max(minw,dim*u);const Y=f=>H*f;
 let s=`<defs><linearGradient id="ag" x1="0" x2="1"><stop offset="0" stop-color="${C}"/><stop offset="1" stop-color="${V}"/></linearGradient><filter id="ag2" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
 const rx=W*.07,ys={in:.03,n1:.11,qkv:.21,rope:.29,at:.38,wo:.48,a1:.56,n2:.64,ff:.73,sg:.82,w2:.9,a2:.975};
 // residual yolu
 rpath=[[rx,Y(ys.in)],[rx,Y(ys.a2)]];
 s+=`<path d="M${cx},${Y(ys.in)} H${rx} V${Y(ys.a1)} H${cx-16} M${rx},${Y(ys.a1)} V${Y(ys.a2)} H${cx-16}" fill="none" stroke="rgba(244,114,182,.55)" stroke-width="2" stroke-dasharray="6 6">${reduce?"":`<animate attributeName="stroke-dashoffset" from="0" to="-24" dur="1.2s" repeatCount="indefinite"/>`}</path>`;
 s+=`<text x="${rx+8}" y="${Y((ys.in+ys.a1)/2)}" font-size="12.5" fill="#f9a8d4" transform="rotate(-90 ${rx+8} ${Y((ys.in+ys.a1)/2)})" text-anchor="middle" dy="-12">residual</text>`;
 // ana yol çizgisi
 s+=`<line x1="${cx}" y1="${Y(ys.in)}" x2="${cx}" y2="${Y(ys.a2)}" stroke="rgba(255,255,255,.12)" stroke-width="2"/>`;
 const pill=(id,y,t,w)=>{s+=`<g class="mat" data-id="${id}" tabindex="0" role="button"><rect x="${cx-w/2}" y="${Y(y)-14}" width="${w}" height="28" rx="14" fill="#0a0d1f" stroke="${sel===id?"#fff":"rgba(255,255,255,.25)"}" stroke-width="${sel===id?2:1}"/><text x="${cx}" y="${Y(y)+5}" text-anchor="middle" font-size="13" font-weight="600" fill="#e8ebf7">${t}</text></g>`};
 const mat=(id,x,y,w,t,dims,colr)=>{const on=sel===id;s+=`<g class="mat" data-id="${id}" tabindex="0" role="button" aria-label="${I[id].ad}"><rect x="${x-w/2}" y="${Y(y)-bh/2}" width="${w}" height="${bh}" rx="8" fill="${colr}" fill-opacity="${on?.55:.22}" stroke="${on?"#fff":colr}" stroke-width="${on?2:1.2}" ${on?'filter="url(#ag2)"':""}/><text x="${x}" y="${Y(y)+5}" text-anchor="middle" font-size="14" font-weight="700" fill="#fff" font-family="Space Grotesk">${t}</text>${W>=480?`<text x="${x}" y="${Y(y)+bh/2+15}" text-anchor="middle" font-size="11.5" fill="#8d95b5">${dims}</text>`:""}</g>`};
 const plus=(y)=>{s+=`<circle cx="${cx}" cy="${Y(y)}" r="15" fill="#0a0d1f" stroke="${P}" stroke-width="2"/><text x="${cx}" y="${Y(y)+6}" text-anchor="middle" font-size="18" font-weight="700" fill="${P}">+</text>`};
 s+=`<text x="${cx}" y="${Y(ys.in)+5}" text-anchor="middle" font-size="13" fill="#a5f3fc" font-weight="600">h  (T × ${fmt(c.d)})</text>`;
 pill("n1",ys.n1,"RMSNorm",Math.min(maxW,180));
 const wq=wOf(c.d),wk=wOf(r.kv),g=10,tot=wq+2*wk+2*g;let x=cx-tot/2;
 mat("wq",x+wq/2,ys.qkv,wq,"W_q",`${c.d}×${c.d}`,V);x+=wq+g;mat("wk",x+wk/2,ys.qkv,wk,"W_k",`${c.d}×${r.kv}`,V);x+=wk+g;mat("wv",x+wk/2,ys.qkv,wk,"W_v",`${c.d}×${r.kv}`,V);
 s+=`<text x="${cx}" y="${Y(ys.rope)+5}" text-anchor="middle" font-size="12.5" fill="#c4b5fd">RoPE(q), RoPE(k) · ${c.H} Q kafası, ${c.Hkv} K/V kafası</text>`;
 pill("at",ys.at,"softmax(QKᵀ/√dh + M)·V",Math.min(maxW,250));
 mat("wo",cx,ys.wo,wOf(c.d),"W_o",`${c.d}×${c.d}`,V);
 plus(ys.a1);
 pill("n2",ys.n2,"RMSNorm",Math.min(maxW,180));
 const wf=wOf(r.f);mat("w1",cx-wf/2-5,ys.ff,wf,"W₁ kapı",`${c.d}×${r.f}`,C);mat("w3",cx+wf/2+5,ys.ff,wf,"W₃ yukarı",`${c.d}×${r.f}`,C);
 pill("sg",ys.sg,`SiLU(·) ⊙ ·   ·   ${fmt(r.f)} nöron`,Math.min(maxW,280));
 mat("w2",cx,ys.w2,wf,"W₂ aşağı",`${r.f}×${c.d}`,C);
 plus(ys.a2);
 s+=`<g id="adots"></g>`;
 svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.innerHTML=s;
 path=[ys.in,ys.n1,ys.qkv,ys.rope,ys.at,ys.wo,ys.a1,ys.n2,ys.ff,ys.sg,ys.w2,ys.a2].map(y=>[cx,Y(y)]);
 const g2=svg.querySelector("#adots");g2.innerHTML=Array.from({length:6},(_,i)=>`<circle r="${i%2?4:5}" fill="${i%2?C:"#fff"}" filter="url(#ag2)"/>`).join("")+Array.from({length:3},()=>`<circle r="4" fill="${P}" filter="url(#ag2)"/>`).join("");
 dots=[...g2.children].map((el,i)=>({el,t:i<6?i/6:(i-6)/3,res:i>=6}));info()}
function along(pts,t){const seg=[];let tot=0;for(let i=1;i<pts.length;i++){const l=Math.hypot(pts[i][0]-pts[i-1][0],pts[i][1]-pts[i-1][1]);seg.push(l);tot+=l}let d=t*tot;for(let i=0;i<seg.length;i++){if(d<=seg[i]){const k=seg[i]?d/seg[i]:0;return[pts[i][0]+(pts[i+1][0]-pts[i][0])*k,pts[i][1]+(pts[i+1][1]-pts[i][1])*k]}d-=seg[i]}return pts.at(-1)}
const rp=()=>rpath;
function tick(){if(isOn(fig)&&dots.length){for(const d of dots){d.t=(d.t+(d.res?.0022:.0028))%1;const q=d.res?along(rp(),d.t):along(path,d.t);d.el.setAttribute("cx",q[0]);d.el.setAttribute("cy",q[1])}}if(!reduce)requestAnimationFrame(tick)}
function info(){const c=cfg(),r=hesap(c,2048),it=items()[sel];
 $("#anatInfo").innerHTML=`<h4 class="gt" style="font-size:21px">${esc(it.ad)}</h4><table class="numtab"><tr><td>şekil</td><td class="n">${esc(it.sh)}</td></tr><tr><td>parametre</td><td class="n">${fmt(it.p)}</td></tr><tr><td>bloğun payı</td><td class="n">${it.p?dec2(100*it.p/r.blok,1)+"%":"—"}</td></tr><tr><td>token başına FLOP</td><td class="n">${it.fl?fmt(it.fl):"—"}</td></tr><tr><td>tüm modelde (× ${c.L})</td><td class="n">${it.p?fmt(it.p*c.L):"—"}</td></tr></table><p style="margin-top:10px">${esc(it.rol)}</p>`;
 $("#anatStats").innerHTML=[[fmt(r.blok),"bloktaki parametre"],[dec2(100*r.att/r.blok,1)+"%","attention payı"],[dec2(100*r.mlp/r.blok,1)+"%","FFN payı"],[dec2((2*r.blok+4*2048*c.d)/1e6,0)+" M","token başına FLOP (T=2.048)"]].map(s=>`<div class="stat"><b>${s[0]}</b><span>${s[1]}</span></div>`).join("")}
svg.addEventListener("click",e=>{const g=e.target.closest(".mat");if(!g)return;sel=g.dataset.id;draw()});
svg.addEventListener("keydown",e=>{const g=e.target.closest(".mat");if(g&&(e.key==="Enter"||e.key===" ")){e.preventDefault();sel=g.dataset.id;draw()}});
function blk(){const c=cfg();$("#anatBlkT").textContent=`Blok ${SEL.blk+1} / ${c.L}`;$("#anatBlkP").textContent=`${c.ad} modelinde ${c.L} bloğun hepsi bu yapıda; ${SEL.blk+1}. bloğun ağırlıkları diğerlerinden farklıdır. 3B yığında başka bir bloğa tıklayarak seçebilirsin.`}
onResize(fig,draw);onModel(()=>{draw();blk()});SEL.subs.push(blk);blk();if(!reduce)requestAnimationFrame(tick);
}

/* ================= Parametre hesaplayıcı ================= */
{
const st={...PRESET["1B"],T:2048,preset:"1B"};
const ids={d:$("#i_d"),L:$("#i_L"),H:$("#i_H"),Hkv:$("#i_Hkv"),V:$("#i_V"),m:$("#i_m"),tied:$("#i_tied"),T:$("#i_T")};
const COLS=[["Embedding","emb",P],["Attention","attT",V],["SwiGLU FFN","mlpT",C],["Çıkış","out",B],["Norm","normT","#64748b"]];
const waffle=$("#waffle");waffle.innerHTML="<i></i>".repeat(200);
$("#legend").innerHTML=COLS.slice(0,4).map(c=>`<span><i style="background:${c[2]};box-shadow:0 0 8px ${c[2]}"></i>${c[0]}</span>`).join("");
let shown=0,anim=0,first=true;
function heads(){const opts=[];for(let h=1;h<=128;h++){if(st.d%h===0&&(st.d/h)%2===0&&st.d/h>=16&&st.d/h<=256)opts.push(h)}
 if(!opts.includes(st.H))st.H=opts.reduce((a,b)=>Math.abs(b-st.H)<Math.abs(a-st.H)?b:a,opts[0]);
 ids.H.innerHTML=opts.map(h=>`<option value="${h}" ${h===st.H?"selected":""}>${h} kafa · dh = ${st.d/h}</option>`).join("");
 const kv=[];for(let k=1;k<=st.H;k++)if(st.H%k===0)kv.push(k);if(!kv.includes(st.Hkv))st.Hkv=kv.reduce((a,b)=>Math.abs(b-st.Hkv)<Math.abs(a-st.Hkv)?b:a,kv[0]);
 ids.Hkv.innerHTML=kv.map(k=>`<option value="${k}" ${k===st.Hkv?"selected":""}>${k}${k===st.H?" (MHA)":k===1?" (MQA)":` (her ${st.H/k} Q'ya 1)`}</option>`).join("")}
function fill(r){r.style.setProperty("--p",(100*(r.value-r.min)/(r.max-r.min))+"%")}
function render(){
 heads();ids.d.value=st.d;ids.L.value=st.L;ids.V.value=st.V;ids.m.value=st.m;ids.tied.checked=st.tied;ids.T.value=Math.log2(st.T);
 $("#o_d").textContent=fmt(st.d);$("#o_L").textContent=st.L;$("#o_T").textContent=fmt(st.T);[ids.d,ids.L,ids.T].forEach(fill);
 document.querySelectorAll("#presets button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.p===st.preset));
 const r=hesap(st,st.T),N=r.N;
 const from=shown,t0=performance.now();cancelAnimationFrame(anim);
 const run=t=>{const k=reduce?1:Math.min(1,(t-t0)/(first?1600:700)),e=1-Math.pow(1-k,3);shown=from+(N-from)*e;$("#totBig").textContent=fmt(shown);if(k<1)anim=requestAnimationFrame(run)};anim=requestAnimationFrame(run);first=false;
 $("#totSub").textContent=`parametre · dh = ${r.dh} · FFN f = ${fmt(r.f)}`;
 $("#totB").textContent=N>=1e9?dec2(N/1e9)+" milyar":dec2(N/1e6,1)+" milyon";
 let sum=0;const acc=COLS.slice(0,4).map(c=>{sum+=r[c[1]];return[c[2],r[c[1]]]});
 const cells=[];acc.forEach(([col,v])=>{const n=Math.round(200*v/sum);for(let i=0;i<n;i++)cells.push(col)});while(cells.length<200)cells.push(C);
 [...waffle.children].forEach((el,i)=>{setTimeout(()=>{el.style.background=cells[i];el.style.boxShadow=`0 0 6px ${cells[i]}66`},reduce?0:i*4)});
 $("#bars").innerHTML=COLS.map(c=>`<div class="bar"><span>${c[0]}</span><span class="tr"><i style="width:${(100*r[c[1]]/N).toFixed(2)}%;background:${c[2]};box-shadow:0 0 12px ${c[2]}"></i></span><span class="v">${fmt(r[c[1]])}</span></div>`).join("");
 const S=[[bytes(N*2),"ağırlık belleği (bf16)"],[bytes(N*.5),"ağırlık belleği (int4)"],[bytes(N*16),"AdamW ile eğitim belleği"],[bytes(r.kvBytes),`KV-cache, T = ${fmt(st.T)} (bf16)`],[dec2(r.fwd/1e9)+" GFLOP","token başına ileri hesap"],[fmt(r.ffnNeuron),"FFN nöronu (L × f)"],[fmt(r.allNeuron),"tüm katman çıkış birimi"],["≈ "+fmt(N/r.ffnNeuron),"FFN nöronu başına parametre"],[fmt(r.blok),"tek bloktaki parametre"]];
 $("#stats").innerHTML=S.map(s=>`<div class="stat"><b>${s[0]}</b><span>${s[1]}</span></div>`).join("")}
$("#presets").addEventListener("click",e=>{const b=e.target.closest("button");if(!b)return;Object.assign(st,PRESET[b.dataset.p],{preset:b.dataset.p});render()});
const custom=()=>{st.preset=""};
ids.d.addEventListener("input",()=>{st.d=+ids.d.value;custom();render()});
ids.L.addEventListener("input",()=>{st.L=+ids.L.value;custom();render()});
ids.H.addEventListener("change",()=>{st.H=+ids.H.value;custom();render()});
ids.Hkv.addEventListener("change",()=>{st.Hkv=+ids.Hkv.value;custom();render()});
ids.V.addEventListener("change",()=>{st.V=+ids.V.value;custom();render()});
ids.m.addEventListener("change",()=>{st.m=ids.m.value;custom();render()});
ids.tied.addEventListener("change",()=>{st.tied=ids.tied.checked;custom();render()});
ids.T.addEventListener("input",()=>{st.T=2**+ids.T.value;render()});
onModel(()=>{Object.assign(st,PRESET[M.key],{preset:M.key});render()});
const res=document.querySelector(".res");let started=false;
const io=new IntersectionObserver(es=>{if(es[0].isIntersecting&&!started){started=true;render()}},{threshold:.15});io.observe(res);
setTimeout(()=>{if(!started){started=true;render()}},4000);
}

/* ================= RMSNorm demo + tablo ================= */
{
const box=$("#rmsSvg").parentNode,svg=$("#rmsSvg");let x=[];
function gen(){x=Array.from({length:8},()=>Math.round(gauss()*(.4+rnd()*2.6)*10)/10)}
function draw(){const W=box.clientWidth,H=box.clientHeight;if(!W)return;const ms=x.reduce((a,v)=>a+v*v,0)/x.length,rms=Math.sqrt(ms+1e-5);const y=x.map(v=>v/rms);
 const mx=Math.max(3,...x.map(Math.abs),...y.map(Math.abs));const half=W/2,bw=(half-30)/x.length,mid=H/2,sc=(H/2-28)/mx;
 let s=`<defs><linearGradient id="rg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${C}"/><stop offset="1" stop-color="${V}"/></linearGradient></defs><text x="10" y="18" font-size="13" fill="#8d95b5">x</text><text x="${half+10}" y="18" font-size="13" fill="#8d95b5">RMSNorm(x)</text>`;
 [0,half].forEach((ox,k)=>{s+=`<line x1="${ox+10}" x2="${ox+half-10}" y1="${mid}" y2="${mid}" stroke="rgba(255,255,255,.15)"/>`;(k?y:x).forEach((v,i)=>{const h=v*sc;s+=`<rect x="${ox+14+i*bw}" width="${bw-4}" y="${h>0?mid-h:mid}" height="${Math.abs(h)}" rx="3" fill="${k?"url(#rg)":"rgba(255,255,255,.35)"}" style="${k?`filter:drop-shadow(0 0 6px ${C}88)`:""}"/>`})});
 svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.innerHTML=s;
 $("#rmsTab").innerHTML=`<tr><th>i</th>${x.map((_,i)=>`<th class="n">${i}</th>`).join("")}</tr><tr><td>xᵢ</td>${x.map(v=>`<td class="n">${dec2(v,1)}</td>`).join("")}</tr><tr><td>xᵢ²</td>${x.map(v=>`<td class="n">${dec2(v*v,2)}</td>`).join("")}</tr><tr><td>xᵢ / rms</td>${y.map(v=>`<td class="n" style="color:var(--c)">${dec2(v,2)}</td>`).join("")}</tr>`;
 $("#rmsTxt").textContent=`mean(x²) = ${dec2(ms,3)} → rms = ${dec2(rms,3)}. Normalize edilmiş vektörün rms'i 1,000.`}
gen();onResize(box,draw);$("#rmsBtn").addEventListener("click",()=>{gen();draw()});
}

/* ================= RoPE demo ================= */
{
const box=$("#ropeSvg").parentNode,svg=$("#ropeSvg"),w=.35,aq=.5,ak=1.6,lq=1,lk=.85;
const im=$("#i_m2"),inn=$("#i_n2");
function draw(){const W=box.clientWidth,H=box.clientHeight;if(!W)return;const m=+im.value,n=+inn.value;
 $("#o_m").textContent=m;$("#o_n").textContent=n;[im,inn].forEach(r=>r.style.setProperty("--p",(100*r.value/r.max)+"%"));
 const R=Math.min(W*.3,H/2-24),cx=Math.min(W*.35,R+30),cy=H/2;const A=aq+m*w,Bk=ak+n*w;const dot=lq*lk*Math.cos(A-Bk);
 const pt=(a,l)=>[cx+Math.cos(a)*R*l,cy-Math.sin(a)*R*l];const[qx,qy]=pt(A,lq),[kx,ky]=pt(Bk,lk);
 let s=`<defs><marker id="ahq" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="${C}"/></marker><marker id="ahk" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="${P}"/></marker></defs>`;
 for(let k=0;k<36;k++){const a=k/36*Math.PI*2,[x1,y1]=pt(a,1),[x2,y2]=pt(a,k%3?.96:.92);s+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="rgba(255,255,255,.18)"/>`}
 s+=`<circle cx="${cx}" cy="${cy}" r="${R}" fill="none" stroke="rgba(139,92,246,.4)" style="filter:drop-shadow(0 0 8px ${V})"/><circle cx="${cx}" cy="${cy}" r="3" fill="#fff"/>`;
 const a0=Math.min(A,Bk),a1=Math.max(A,Bk),big=(a1-a0)%(2*Math.PI)>Math.PI?1:0;const p0=pt(a0,.32),p1=pt(a1,.32);
 s+=`<path d="M${p0[0]},${p0[1]} A${R*.32},${R*.32} 0 ${big} 0 ${p1[0]},${p1[1]}" fill="none" stroke="${V}" stroke-width="2.5"/>`;
 s+=`<line x1="${cx}" y1="${cy}" x2="${qx}" y2="${qy}" stroke="${C}" stroke-width="4" marker-end="url(#ahq)" style="filter:drop-shadow(0 0 6px ${C})"/><line x1="${cx}" y1="${cy}" x2="${kx}" y2="${ky}" stroke="${P}" stroke-width="4" marker-end="url(#ahk)" style="filter:drop-shadow(0 0 6px ${P})"/>`;
 const[lqx,lqy]=pt(A,1.17),[lkx,lky]=pt(Bk,.45);
 s+=`<text x="${lqx}" y="${lqy+5}" text-anchor="middle" font-size="16" font-weight="700" fill="${C}">q</text><circle cx="${lkx}" cy="${lky}" r="11" fill="#0a0d1f"/><text x="${lkx}" y="${lky+5}" text-anchor="middle" font-size="15" font-weight="700" fill="${P}">k</text>`;
 const tx=cx+R+36;if(tx+120<W){s+=`<text x="${tx}" y="${cy-48}" font-size="14" fill="#8d95b5">açı farkı</text><text x="${tx}" y="${cy-22}" font-size="20" font-weight="700" fill="#c4b5fd" font-family="Space Grotesk">${dec2(((A-Bk)*180/Math.PI),0)}°</text><text x="${tx}" y="${cy+4}" font-size="14" fill="#8d95b5">m − n</text><text x="${tx}" y="${cy+32}" font-size="28" font-weight="700" fill="#e8ebf7" font-family="Space Grotesk">${m-n}</text><text x="${tx}" y="${cy+58}" font-size="14" fill="#8d95b5">q·k</text><text x="${tx}" y="${cy+86}" font-size="28" font-weight="700" fill="${C}" font-family="Space Grotesk">${dec2(dot,3)}</text>`}
 svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.innerHTML=s;
 $("#ropeTxt").textContent=`q ${dec2(A*180/Math.PI%360,0)}°, k ${dec2(Bk*180/Math.PI%360,0)}° döndü (ω = ${w} rad/konum). m − n = ${m-n} → q·k = ${dec2(dot,3)}. "İkisini kaydır" ile açılar değişir, fark ve skor aynı kalır.`}
[im,inn].forEach(r=>r.addEventListener("input",draw));
$("#ropeShift").addEventListener("click",()=>{if(+im.value+5>40||+inn.value+5>40){im.value=Math.max(0,im.value-30);inn.value=Math.max(0,inn.value-30)}else{im.value=+im.value+5;inn.value=+inn.value+5}draw()});
onResize(box,draw);
}

/* ================= Frekans saatleri + tablo ================= */
{
const box=$("#clockSvg").parentNode,svg=$("#clockSvg");let m=0,run=!reduce,hands=[],last=0;const dh=16,theta=10000;vio.observe(box);
const om=Array.from({length:dh/2},(_,i)=>Math.pow(theta,-2*i/dh)),om3=Array.from({length:dh/2},(_,i)=>Math.pow(5e5,-2*i/dh));
$("#clockTab").innerHTML=`<tr><th>çift i</th>${om.map((_,i)=>`<th class="n">${i}</th>`).join("")}</tr><tr><td>ωᵢ (θ=10⁴)</td>${om.map(w=>`<td class="n">${w<.01?w.toExponential(1).replace(".",","):dec2(w,3)}</td>`).join("")}</tr><tr><td>tam tur (konum)</td>${om.map(w=>`<td class="n">${fmt(2*Math.PI/w)}</td>`).join("")}</tr><tr><td>tam tur, θ=5·10⁵</td>${om3.map(w=>`<td class="n" style="color:var(--p)">${fmt(2*Math.PI/w)}</td>`).join("")}</tr>`;
function draw(){const W=box.clientWidth,H=box.clientHeight;if(!W)return;const n=om.length,cols=W<520?4:8,rows=Math.ceil(n/cols),cw=W/cols,ch=H/rows,r=Math.min(cw,ch)/2-22;
 let s="";om.forEach((w,i)=>{const cx=cw*(i%cols)+cw/2,cy=ch*Math.floor(i/cols)+ch/2-8,hue=i/(n-1);const col=hue<.5?C:hue<.8?V:P;
  for(let k=0;k<12;k++){const a=k/12*Math.PI*2;s+=`<line x1="${cx+Math.sin(a)*r*.86}" y1="${cy-Math.cos(a)*r*.86}" x2="${cx+Math.sin(a)*r}" y2="${cy-Math.cos(a)*r}" stroke="rgba(255,255,255,.2)"/>`}
  s+=`<circle cx="${cx}" cy="${cy}" r="${r}" fill="rgba(255,255,255,.03)" stroke="${col}" stroke-opacity=".45" stroke-width="2" style="filter:drop-shadow(0 0 6px ${col}66)"/><line class="hd" x1="${cx}" y1="${cy}" x2="${cx}" y2="${cy-r}" stroke="${col}" stroke-width="3" stroke-linecap="round" style="filter:drop-shadow(0 0 5px ${col})"/><circle cx="${cx}" cy="${cy}" r="3" fill="#fff"/><text x="${cx}" y="${cy+r+17}" text-anchor="middle" font-size="13" fill="#8d95b5">i=${i}</text>`});
 svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.innerHTML=s;hands=[...svg.querySelectorAll(".hd")].map(el=>({el,cx:+el.getAttribute("x1"),cy:+el.getAttribute("y1"),r}));upd()}
function upd(){hands.forEach((h,i)=>{const a=m*om[i];h.el.setAttribute("x2",h.cx+Math.sin(a)*h.r);h.el.setAttribute("y2",h.cy-Math.cos(a)*h.r)});$("#clockM").textContent=Math.floor(m)}
function loop(t){if(run&&isOn(box)){m+=(last?Math.min(50,t-last):0)/180;upd()}last=t;requestAnimationFrame(loop)}
onResize(box,draw);requestAnimationFrame(loop);
const b=$("#clockBtn");b.textContent=run?"Durdur":"Oynat";b.addEventListener("click",()=>{run=!run;b.textContent=run?"Durdur":"Oynat"});
}

/* ================= Attention ısı haritası ================= */
{
const box=$("#attSvg").parentNode,svg=$("#attSvg");const T=["Bir","var","mış","bir","yok","muş","ev","vel"],n=T.length,dh=8;let A=[];let scan=0;
function gen(){const q=T.map(()=>Array.from({length:dh},gauss)),k=T.map(()=>Array.from({length:dh},gauss));
 A=q.map((qi,i)=>{const sc=k.map((kj,j)=>j>i?-Infinity:qi.reduce((a,v,d)=>a+v*kj[d],0)/Math.sqrt(dh)*1.6);const mx=Math.max(...sc.filter(isFinite));const e=sc.map(v=>isFinite(v)?Math.exp(v-mx):0);const z=e.reduce((a,b)=>a+b,0);return e.map(v=>v/z)})}
function draw(){const W=box.clientWidth,H=box.clientHeight;if(!W)return;const lab=46,c=Math.min((W-lab-8)/n,(H-lab-8)/n),ox=lab+(W-lab-c*n)/2-lab/2,oy=lab;
 let s=`<defs><pattern id="hx" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="rgba(255,255,255,.12)" stroke-width="2"/></pattern></defs>`;
 T.forEach((t,i)=>{s+=`<text x="${ox-8}" y="${oy+i*c+c/2+5}" text-anchor="end" font-size="13" fill="#cfd5ec">${t}</text><text x="${ox+i*c+c/2}" y="${oy-10}" text-anchor="middle" font-size="13" fill="#cfd5ec">${t}</text>`});
 A.forEach((row,i)=>row.forEach((v,j)=>{const x=ox+j*c,y=oy+i*c;s+=j>i?`<rect x="${x+1}" y="${y+1}" width="${c-2}" height="${c-2}" fill="url(#hx)"/>`:`<rect x="${x+1.5}" y="${y+1.5}" width="${c-3}" height="${c-3}" rx="4" fill="${v>.5?P:v>.2?V:C}" fill-opacity="${(.1+.9*v).toFixed(3)}" style="${v>.4?`filter:drop-shadow(0 0 6px ${P})`:""}"/>${c>34?`<text x="${x+c/2}" y="${y+c/2+4}" text-anchor="middle" font-size="10.5" fill="#fff" opacity=".8">${Math.round(v*100)}</text>`:""}`}));
 s+=`<rect id="scan" x="${ox}" y="${oy}" width="${c*n}" height="${c}" fill="none" stroke="#fff" stroke-opacity=".8" stroke-width="2" rx="5"/>`;
 svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.innerHTML=s;svg.dataset.c=c;svg.dataset.oy=oy}
gen();onResize(box,draw);$("#attBtn").addEventListener("click",()=>{gen();draw()});
if(!reduce)setInterval(()=>{scan=(scan+1)%n;const r=svg.querySelector("#scan");if(r)r.setAttribute("y",+svg.dataset.oy+scan*+svg.dataset.c)},900);
}

/* ================= GQA ================= */
{
const box=$("#gqaSvg").parentNode,svg=$("#gqaSvg");let ph=0;
function draw(){const W=box.clientWidth,H=box.clientHeight;if(!W)return;const qy=H*.2,ky=H*.78,mx=14,qs=(W-2*mx)/32,ks=(W-2*mx)/8;let s="";
 for(let g=0;g<8;g++){const kx=mx+ks*g+ks/2;for(let j=0;j<4;j++){const i=g*4+j,qx=mx+qs*i+qs/2;s+=`<path class="lk" data-g="${g}" d="M${qx},${qy+8} C${qx},${(qy+ky)/2} ${kx},${(qy+ky)/2} ${kx},${ky-14}" fill="none" stroke="rgba(255,255,255,.18)" stroke-width="1.5"/>`}}
 for(let i=0;i<32;i++){const qx=mx+qs*i+qs/2;s+=`<circle class="q" data-g="${Math.floor(i/4)}" cx="${qx}" cy="${qy}" r="${Math.min(7,qs*.38)}" fill="${C}" fill-opacity=".55"/>`}
 for(let g=0;g<8;g++){const kx=mx+ks*g+ks/2,w=Math.min(ks-8,54);s+=`<rect class="kv" data-g="${g}" x="${kx-w/2}" y="${ky-14}" width="${w}" height="28" rx="8" fill="${V}" fill-opacity=".55"/><text x="${kx}" y="${ky+5}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#fff">KV${g+1}</text>`}
 s+=`<text x="${mx}" y="${qy-18}" font-size="13" fill="#8d95b5">32 sorgu kafası (Q)</text><text x="${mx}" y="${ky+40}" font-size="13" fill="#8d95b5">8 paylaşılan K/V kafası · KV-cache 4 kat küçük</text>`;
 svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.innerHTML=s;hl()}
function hl(){svg.querySelectorAll("[data-g]").forEach(el=>{const on=+el.dataset.g===ph;if(el.classList.contains("lk")){el.setAttribute("stroke",on?P:"rgba(255,255,255,.18)");el.setAttribute("stroke-width",on?2.5:1.5);el.style.filter=on?`drop-shadow(0 0 5px ${P})`:""}else{el.setAttribute("fill-opacity",on?1:.55);el.style.filter=on?`drop-shadow(0 0 8px ${el.classList.contains("q")?C:V})`:""}})}
onResize(box,draw);if(!reduce)setInterval(()=>{ph=(ph+1)%8;hl()},1100);
}

/* ================= SiLU grafiği ================= */
{
const box=$("#siluSvg").parentNode,svg=$("#siluSvg");let done=false;
function draw(){const W=box.clientWidth,H=box.clientHeight;if(!W)return;const m={l:34,r:12,t:12,b:28},x0=-5,x1=5,y0=-1,y1=5;
 const X=v=>m.l+(v-x0)/(x1-x0)*(W-m.l-m.r),Y=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);
 const path=f=>{let d="";for(let i=0;i<=120;i++){const v=x0+(x1-x0)*i/120;d+=(i?"L":"M")+X(v).toFixed(1)+","+Y(f(v)).toFixed(1)}return d};
 let s="";[-4,-2,0,2,4].forEach(v=>s+=`<line x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${H-m.b}" stroke="rgba(255,255,255,.06)"/><text x="${X(v)}" y="${H-8}" text-anchor="middle" font-size="13" fill="#8d95b5">${v}</text>`);
 [0,2,4].forEach(v=>s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="rgba(255,255,255,${v?.06:.2})"/><text x="${m.l-8}" y="${Y(v)+4}" text-anchor="end" font-size="13" fill="#8d95b5">${v}</text>`);
 const anim=!reduce&&!done;
 s+=`<path d="${path(v=>Math.max(0,v))}" fill="none" stroke="rgba(255,255,255,.45)" stroke-width="2" stroke-dasharray="5 5"/><path d="${path(v=>v/(1+Math.exp(-v)))}" fill="none" stroke="${C}" stroke-width="3.5" pathLength="1" stroke-dasharray="1" stroke-dashoffset="${anim?1:0}" style="filter:drop-shadow(0 0 8px ${C})">${anim?`<animate id="siluA" attributeName="stroke-dashoffset" from="1" to="0" dur="1.6s" fill="freeze" begin="indefinite"/>`:""}</path>`;
 s+=`<circle cx="${X(-1.278)}" cy="${Y(-.278)}" r="4" fill="${P}"/><text x="${X(2.4)}" y="${Y(3.6)}" font-size="14" font-weight="700" fill="${C}">SiLU</text><text x="${X(-4.6)}" y="${Y(.35)}" font-size="13" fill="#8d95b5">ReLU (kesikli)</text><text x="${X(-1.28)}" y="${Y(-.6)}" font-size="13" fill="#cfd5ec">min ≈ −0,28 (x ≈ −1,28)</text>`;
 svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.innerHTML=s}
onResize(box,draw);
const play=()=>{if(done)return;const a=svg.querySelector("#siluA");if(a&&a.beginElement)a.beginElement();setTimeout(()=>done=true,1700)};
const io=new IntersectionObserver(es=>{if(es[0].isIntersecting)play()},{threshold:.4});io.observe(box);setTimeout(play,9000);
}

/* ================= Canlı attention (kendi cümlen) ================= */
{
const inp=$("#liveIn"),box=$("#liveSvg").parentNode,svg=$("#liveSvg");const D=16;let toks=[],A=[],q=0;
const Wr=mkRng(424242),Wq=Array.from({length:D},()=>Array.from({length:D},()=>Wr.g()/Math.sqrt(D))),Wk=Array.from({length:D},()=>Array.from({length:D},()=>Wr.g()/Math.sqrt(D)));
const emb=t=>{const g=mkRng(hash(t.toLocaleLowerCase("tr"))).g;return Array.from({length:D},()=>g()*1.6)};
const mul=(x,W)=>W[0].map((_,j)=>x.reduce((a,v,i)=>a+v*W[i][j],0));
const rope=(x,m)=>{const o=x.slice();for(let i=0;i<D/2;i++){const w=Math.pow(1e4,-2*i/D),c=Math.cos(m*w),s=Math.sin(m*w);o[2*i]=x[2*i]*c-x[2*i+1]*s;o[2*i+1]=x[2*i]*s+x[2*i+1]*c}return o};
function compute(){toks=inp.value.trim().split(/\s+/).filter(Boolean).slice(0,12);if(!toks.length)toks=["…"];if(q>=toks.length)q=toks.length-1;
 const useR=$("#liveRope").checked,useM=$("#liveMask").checked,useS=$("#liveScale").checked;
 const E=toks.map(emb),Q=E.map((e,i)=>{const v=mul(e,Wq);return useR?rope(v,i):v}),K=E.map((e,i)=>{const v=mul(e,Wk);return useR?rope(v,i):v});
 A=Q.map((qi,i)=>{const sc=K.map((kj,j)=>useM&&j>i?-Infinity:qi.reduce((a,v,d)=>a+v*kj[d],0)/(useS?Math.sqrt(D):1));const mx=Math.max(...sc.filter(isFinite));const e=sc.map(v=>isFinite(v)?Math.exp(v-mx):0);const z=e.reduce((a,b)=>a+b,0);return e.map(v=>v/z)});
 const dup=toks.some((t,i)=>toks.findIndex(u=>u.toLocaleLowerCase("tr")===t.toLocaleLowerCase("tr"))!==i);
 const notes=[];if(!useM)notes.push("Maske kapalı: tokenlar gelecekteki kelimelere de bakıyor; eğitimde bu, cevabı kopya çekmek olur.");
 if(!useR&&dup)notes.push("RoPE kapalı: aynı kelimenin iki kopyası konumdan habersiz, bu yüzden aynı puanı alıyor.");
 if(useR&&dup)notes.push("RoPE açık: aynı kelimenin kopyaları farklı konumda olduğu için farklı puan alıyor.");
 if(!useS)notes.push("√dh'ye bölme kapalı: puanlar büyüyor, softmax tek bir token'a yığılmaya başlıyor.");
 $("#liveNote").textContent=notes.join(" ");render()}
function render(){$("#liveChips").innerHTML=toks.map((t,i)=>`<button data-i="${i}" aria-pressed="${i===q}">${esc(t)}</button>`).join("");
 const row=A[q]||[],useM=$("#liveMask").checked;
 $("#liveBars").innerHTML=toks.map((t,j)=>{const masked=useM&&j>q;return`<div class="wbar${masked?" mask":""}"><span>${esc(t)}</span><span class="tr"><i style="width:${masked?0:(100*row[j]).toFixed(1)}%"></i></span><span>${masked?"maske":dec2(100*row[j],1)+"%"}</span></div>`}).join("");draw()}
function draw(){const W=box.clientWidth,H=box.clientHeight;if(!W)return;const n=toks.length,lab=Math.min(84,W*.2),c=Math.min((W-lab-8)/n,(H-lab-8)/n),ox=lab,oy=lab;
 let s=`<defs><pattern id="hx2" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="rgba(255,255,255,.12)" stroke-width="2"/></pattern></defs>`;
 const short=t=>t.length>9?t.slice(0,8)+"…":t;
 toks.forEach((t,i)=>{s+=`<text x="${ox-8}" y="${oy+i*c+c/2+5}" text-anchor="end" font-size="12.5" fill="${i===q?C:"#cfd5ec"}" font-weight="${i===q?700:400}">${esc(short(t))}</text><text transform="translate(${ox+i*c+c/2},${oy-8}) rotate(-50)" font-size="12.5" fill="#cfd5ec">${esc(short(t))}</text>`});
 const useM=$("#liveMask").checked;
 A.forEach((row,i)=>row.forEach((v,j)=>{const x=ox+j*c,y=oy+i*c;s+=useM&&j>i?`<rect x="${x+1}" y="${y+1}" width="${c-2}" height="${c-2}" fill="url(#hx2)"/>`:`<rect class="lc" data-i="${i}" x="${x+1.5}" y="${y+1.5}" width="${c-3}" height="${c-3}" rx="4" fill="${v>.5?P:v>.2?V:C}" fill-opacity="${(.08+.92*v).toFixed(3)}" style="cursor:pointer"/>`}));
 s+=`<rect x="${ox}" y="${oy+q*c}" width="${c*n}" height="${c}" fill="none" stroke="#fff" stroke-width="2" rx="5" style="filter:drop-shadow(0 0 6px ${C})"/>`;
 svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.innerHTML=s}
$("#liveChips").addEventListener("click",e=>{const b=e.target.closest("button");if(!b)return;q=+b.dataset.i;render()});
svg.addEventListener("click",e=>{const r=e.target.closest(".lc");if(!r)return;q=+r.dataset.i;render()});
inp.addEventListener("input",compute);["#liveRope","#liveMask","#liveScale"].forEach(s=>$(s).addEventListener("change",compute));
q=5;onResize(box,draw);compute();
}

/* ================= KV-cache ================= */
{
const box=$("#kvSvg").parentNode,svg=$("#kvSvg");vio.observe(box);const TK=["Bir","varmış","bir","yokmuş","evvel","zaman","içinde","kalbur","saman","içinde","develer","tellal"];const P0=3;let t=P0,play=0,flash=-1;
function draw(){const W=box.clientWidth,H=box.clientHeight;if(!W)return;const n=TK.length,rowH=(H-58)/n,tw=Math.min(110,W*.22),gw=(W-tw-60)/2,cells=8,cw=gw/cells;
 let s=`<text x="0" y="16" font-size="13" fill="#8d95b5">token</text><text x="${tw+20}" y="16" font-size="13" fill="${V}" font-weight="600">K önbelleği</text><text x="${tw+40+gw}" y="16" font-size="13" fill="${C}" font-weight="600">V önbelleği</text>`;
 for(let i=0;i<n;i++){const y=30+i*rowH,on=i<t,isNew=i===t-1&&t>P0,pr=i<P0;
  s+=`<text x="0" y="${y+rowH*.68}" font-size="13.5" font-weight="${isNew?700:500}" fill="${!on?"#3b4262":isNew?"#fff":pr?"#a5f3fc":"#cfd5ec"}">${esc(TK[i])}${pr&&on?" ·":""}</text>`;
  for(let k=0;k<cells;k++){[[tw+20,V],[tw+40+gw,C]].forEach(([ox,colr])=>{s+=`<rect x="${ox+k*cw+1}" y="${y+2}" width="${cw-2}" height="${rowH-4}" rx="3" fill="${on?colr:"rgba(255,255,255,.04)"}" fill-opacity="${on?(isNew?.9:.35+.25*Math.sin(i+k)):1}" ${isNew&&flash===i?`style="filter:drop-shadow(0 0 8px ${colr})"`:""}/>`})}}
 if(t>P0){const yq=30+(t-1)*rowH+rowH/2;for(let i=0;i<t;i++){const y=30+i*rowH+rowH/2;s+=`<path d="M${tw+10},${yq} C${tw-6},${(yq+y)/2} ${tw-6},${(yq+y)/2} ${tw+18},${y}" fill="none" stroke="${P}" stroke-opacity="${.25+.6*(i===t-1)}" stroke-width="1.5"/>`}}
 s+=`<text x="${W}" y="${H-4}" text-anchor="end" font-size="12" fill="#8d95b5">her satır = bir konum, her kutu = bir K/V kafası (8)</text>`;
 svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.innerHTML=s;stats()}
function stats(){const c=cfg(),r=hesap(c,1),per=2*c.L*c.Hkv*r.dh*2;let a=0,b=0;for(let s=1;s<=t;s++){a+=s;b+=s*s}const mx=TK.length*(TK.length+1)*(2*TK.length+1)/6;
 $("#kvA").style.width=(100*a/mx)+"%";$("#kvB").style.width=(100*b/mx)+"%";$("#kvAv").textContent=fmt(a);$("#kvBv").textContent=fmt(b);
 $("#kvStats").innerHTML=[[t+" / "+TK.length,"dizi uzunluğu T"],[bytes(per),`token başına KV (${M.key})`],[bytes(per*t),`şu anki önbellek`],[bytes(per*8192),"8.192 token'da"]].map(s=>`<div class="stat"><b>${s[0]}</b><span>${s[1]}</span></div>`).join("")}
function step(){if(t>=TK.length){t=P0}else t++;flash=t-1;draw()}
$("#kvStep").addEventListener("click",()=>{clearInterval(play);play=0;$("#kvPlay").textContent="Oynat";step()});
$("#kvReset").addEventListener("click",()=>{clearInterval(play);play=0;$("#kvPlay").textContent="Oynat";t=P0;draw()});
$("#kvPlay").addEventListener("click",e=>{if(play){clearInterval(play);play=0;e.target.textContent="Oynat"}else{play=setInterval(step,reduce?400:900);e.target.textContent="Durdur"}});
onResize(box,draw);onModel(stats);
if(!reduce){const io=new IntersectionObserver(es=>{if(es[0].isIntersecting&&!play){play=setInterval(()=>{step();if(t===TK.length){clearInterval(play);play=0;$("#kvPlay").textContent="Oynat"}},900);$("#kvPlay").textContent="Durdur";io.disconnect()}},{threshold:.4});io.observe(box)}
}

/* ================= Eğitim döngüsü ================= */
{
const box=$("#cycSvg").parentNode,svg=$("#cycSvg");vio.observe(box);let k=0,ph=0,run=!reduce,adim=0,last=0;
const S=[["Veri","x, y = veri.batch(\"train\")","Bellekten eşlenmiş (memmap) token dosyasından rastgele pencereler alınır. y, x'in bir sağa kaydırılmışıdır: her konum bir sonrakini tahmin eder."],
 ["İleri geçiş","_, loss = model(x, y)","bf16 autocast altında ileri geçiş. Logit'ler fp32'ye çevrilir, kararlılık için."],
 ["Kayıp","F.cross_entropy(logits, y)","Doğru token'ın −log olasılığının ortalaması. Başlangıçta ≈ ln V; mükemmel tahminde 0."],
 ["Geri yayılım","scaler.scale(loss).backward()","Zincir kuralıyla her parametrenin gradyanı. Gradyan biriktirmede k mikro-batch'in gradyanı toplanır, kayıp 1/k ile ölçeklenir."],
 ["Gradyan kırpma","clip_grad_norm_(params, 1.0)","‖g‖ > 1 ise g ← g/‖g‖. Ani patlamaları keser; log'daki |g| değeri budur."],
 ["AdamW","optimizer.step()","Her parametre için momentum m ve ölçek v güncellenir, ağırlık azaltma uygulanır. Fused çekirdek GPU'da tek geçişte yapar."],
 ["Öğrenme oranı","lr = ogrenme_orani(adim, a)","Bir sonraki adımın η'sı: ilk 200 adım doğrusal ısınma, sonra kosinüsle 3·10⁻⁵'e iniş."]];
const loss=[[0,5.7901],[5,4.1462],[10,2.3875],[15,1.5421],[20,1.158],[25,1.0071],[30,.9272]];
const li=x=>{for(let i=1;i<loss.length;i++)if(x<=loss[i][0]){const a=loss[i-1],b=loss[i],u=(x-a[0])/(b[0]-a[0]);return a[1]+(b[1]-a[1])*u}return loss.at(-1)[1]};
let nodes=[];
function draw(){const W=box.clientWidth,H=box.clientHeight;if(!W)return;const cx=W/2,cy=H/2,R=Math.min(W,H)/2-56;nodes=S.map((_,i)=>{const a=-Math.PI/2+i/S.length*Math.PI*2;return[cx+Math.cos(a)*R,cy+Math.sin(a)*R,a]});
 let s=`<defs><linearGradient id="cg" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stop-color="${C}"/><stop offset=".5" stop-color="${V}"/><stop offset="1" stop-color="${P}"/></linearGradient><filter id="cgl" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
 s+=`<circle cx="${cx}" cy="${cy}" r="${R}" fill="none" stroke="url(#cg)" stroke-width="3" stroke-dasharray="3 10" opacity=".8">${reduce?"":`<animateTransform attributeName="transform" type="rotate" from="0 ${cx} ${cy}" to="360 ${cx} ${cy}" dur="40s" repeatCount="indefinite"/>`}</circle><circle cx="${cx}" cy="${cy}" r="${R*.62}" fill="rgba(139,92,246,.06)" stroke="rgba(255,255,255,.08)"/>`;
 nodes.forEach(([x,y,a],i)=>{const lx=x+Math.cos(a)*34,ly=y+Math.sin(a)*34+5,an=Math.abs(Math.cos(a))<.3?"middle":Math.cos(a)>0?"start":"end";
  s+=`<g class="cn" data-i="${i}" style="cursor:pointer" tabindex="0" role="button" aria-label="${S[i][0]}"><circle cx="${x}" cy="${y}" r="20" fill="#0a0d1f" stroke="url(#cg)" stroke-width="2"/><circle class="ring" cx="${x}" cy="${y}" r="27" fill="none" stroke="${C}" stroke-width="2" opacity="0" filter="url(#cgl)"/><text x="${x}" y="${y+5}" text-anchor="middle" font-size="14" font-weight="700" fill="#e8ebf7" font-family="Space Grotesk">${i+1}</text><text x="${lx}" y="${ly}" text-anchor="${an}" font-size="13.5" font-weight="600" fill="#cfd5ec">${S[i][0]}</text></g>`});
 s+=`<text id="cA" x="${cx}" y="${cy-18}" text-anchor="middle" font-size="13" fill="#8d95b5">adım</text><text id="cS" x="${cx}" y="${cy+14}" text-anchor="middle" font-size="34" font-weight="700" fill="#fff" font-family="Space Grotesk">0</text><text id="cL" x="${cx}" y="${cy+40}" text-anchor="middle" font-size="13.5" fill="${C}">kayıp 5,79</text><circle id="cd" r="8" fill="#fff" filter="url(#cgl)"/>`;
 svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.innerHTML=s;mark();pos()}
function mark(){svg.querySelectorAll(".cn").forEach((g,i)=>g.querySelector(".ring").setAttribute("opacity",i===k?1:0));
 const s=S[k];$("#cycInfo").innerHTML=`<h4 class="gt" style="font-size:22px">${k+1}. ${s[0]}</h4><div class="eq" style="font-size:14px">${esc(s[1])}</div><p>${esc(s[2])}</p>`}
function pos(){if(!nodes.length)return;const a=nodes[k],b=nodes[(k+1)%nodes.length],W=box.clientWidth,H=box.clientHeight,cx=W/2,cy=H/2,R=Math.min(W,H)/2-56;const a0=a[2],a1=a0+Math.PI*2/S.length*ph;const d=svg.querySelector("#cd");if(d){d.setAttribute("cx",cx+Math.cos(a1)*R);d.setAttribute("cy",cy+Math.sin(a1)*R)}
 const st=svg.querySelector("#cS"),lt=svg.querySelector("#cL");if(st){st.textContent=adim;lt.textContent=`kayıp ${dec2(li(Math.min(30,adim)),2)} · CI verisi`}}
function loop(t){if(run&&isOn(box)){ph+=(last?Math.min(50,t-last):0)/900;if(ph>=1){ph=0;k=(k+1)%S.length;if(k===0){adim=adim>=30?0:adim+1}mark()}pos()}last=t;requestAnimationFrame(loop)}
svg.addEventListener("click",e=>{const g=e.target.closest(".cn");if(!g)return;k=+g.dataset.i;ph=0;mark();pos()});
svg.addEventListener("keydown",e=>{const g=e.target.closest(".cn");if(g&&(e.key==="Enter"||e.key===" ")){e.preventDefault();k=+g.dataset.i;ph=0;mark();pos()}});
const b=$("#cycBtn");b.textContent=run?"Durdur":"Oynat";b.addEventListener("click",()=>{run=!run;b.textContent=run?"Durdur":"Oynat"});
onResize(box,draw);requestAnimationFrame(loop);
}

/* ================= Kayıp ve LR grafikleri ================= */
function lineChart(svg,series,o){const box=svg.parentNode;let played=false;
 function draw(){const W=box.clientWidth,H=box.clientHeight;if(!W)return;const m={l:52,r:W<520?14:62,t:16,b:34};
  const X=v=>m.l+(v-o.x0)/(o.x1-o.x0)*(W-m.l-m.r),Y=v=>H-m.b-(v-o.y0)/(o.y1-o.y0)*(H-m.t-m.b);
  let s=`<defs><linearGradient id="ar${o.id}" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${C}" stop-opacity=".35"/><stop offset="1" stop-color="${V}" stop-opacity="0"/></linearGradient></defs>`;
  o.yt.forEach(v=>s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="rgba(255,255,255,.07)"/><text x="${m.l-8}" y="${Y(v)+4}" text-anchor="end" font-size="13" fill="#8d95b5">${o.yf(v)}</text>`);
  o.xt.filter((v,i)=>W>=520||i%2===0).forEach(v=>s+=`<text x="${X(v)}" y="${H-10}" text-anchor="middle" font-size="13" fill="#8d95b5">${o.xf(v)}</text>`);
  series.forEach(se=>{const d=se.p.map((p,i)=>(i?"L":"M")+X(p[0]).toFixed(1)+","+Y(p[1]).toFixed(1)).join("");
   if(se.area)s+=`<path d="${d}L${X(se.p.at(-1)[0])},${Y(o.y0)}L${X(se.p[0][0])},${Y(o.y0)}Z" fill="url(#ar${o.id})"/>`;
   const anim=!reduce&&!se.dash&&!played;
   s+=`<path d="${d}" fill="none" stroke="${se.c}" stroke-width="${se.w||3}" ${se.dash?'stroke-dasharray="6 5"':`pathLength="1" stroke-dasharray="1" stroke-dashoffset="${anim?1:0}"`} style="filter:drop-shadow(0 0 6px ${se.c})">${anim?`<animate class="la" attributeName="stroke-dashoffset" from="1" to="0" dur="1.8s" fill="freeze" begin="indefinite"/>`:""}</path>`;
   if(se.dots)se.p.forEach(p=>s+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4.5" fill="#05060f" stroke="${se.c}" stroke-width="2.5"><title>adım ${p[0]}: ${dec2(p[1],4)}</title></circle>`)});
  if(o.note)s+=o.note(X,Y,W);
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.innerHTML=s}
 onResize(box,draw);
 const play=()=>{if(played)return;svg.querySelectorAll(".la").forEach(a=>a.beginElement&&a.beginElement());setTimeout(()=>played=true,1900)};
 const io=new IntersectionObserver(es=>{if(es[0].isIntersecting)play()},{threshold:.4});io.observe(box);setTimeout(play,9000)}
{
const tr=[[0,5.7901],[5,4.1462],[10,2.3875],[15,1.5421],[20,1.158],[25,1.0071],[30,.9272]],va=[[10,2.3887],[20,1.2315],[30,.9412]];
lineChart($("#lossSvg"),[{p:tr,c:C,dots:true,area:true},{p:va,c:"#c4b5fd",dots:true,dash:true,w:2}],
 {id:"l",x0:0,x1:30,y0:0,y1:6,xt:[0,5,10,15,20,25,30],yt:[0,2,4,6],xf:v=>v,yf:v=>dec2(v,1),
  note:(X,Y,W)=>W>=380?`<text x="${X(10.5)}" y="${Y(5.2)}" font-size="22" font-weight="700" fill="#e8ebf7" font-family="Space Grotesk">5,79 → 0,93</text><text x="${X(10.5)}" y="${Y(5.2)+20}" font-size="13" fill="#8d95b5">30 adımda · perplexity 2,6</text>`:""});
const lr=[];for(let s=0;s<=5000;s+=25){let v;if(s<200)v=3e-4*(s+1)/200;else{const r=(s-200)/4800;v=3e-5+.5*(1+Math.cos(Math.PI*r))*(3e-4-3e-5)}lr.push([s,v*1e4])}
lineChart($("#lrSvg"),[{p:lr,c:V,area:true}],{id:"r",x0:0,x1:5000,y0:0,y1:3.2,xt:[0,1000,2000,3000,4000,5000],yt:[0,1,2,3],xf:v=>fmt(v),yf:v=>v?v+"e-4":"0",
 note:(X,Y,W)=>W>=380?`<text x="${X(260)}" y="${Y(3.05)}" font-size="13" fill="#cfd5ec">ısınma</text><text x="${X(2300)}" y="${Y(2.15)}" font-size="13" fill="#cfd5ec">kosinüs iniş</text>`:""});
}

/* ================= GPU bellek ================= */
{
const ib=$("#i_mb"),it=$("#i_mt"),ick=$("#i_ck");const G=2**30;
const PARTS=[["Ağırlık (fp32)","#a5b4fc"],["Gradyan","#8b5cf6"],["Adam m + v","#6d28d9"],["Aktivasyon","#22d3ee"],["Logit'ler","#f472b6"],["Ağırlık (bf16)","#a5b4fc"],["KV-cache","#22d3ee"]];
$("#memLeg").innerHTML=PARTS.slice(0,5).map(p=>`<span><i style="background:${p[1]}"></i>${p[0]}</span>`).join("")+`<span><i style="background:#fff"></i>GPU sınırı (24 / 40 / 80 GiB)</span>`;
function render(){const c=cfg(),r=hesap(c,1),N=r.N,Bt=2**+ib.value,T=2**+it.value;$("#o_mb").textContent=Bt;$("#o_mt").textContent=fmt(T);[ib,it].forEach(x=>x.style.setProperty("--p",(100*(x.value-x.min)/(x.max-x.min))+"%"));
 const sbd=T*Bt*c.d,act=ick.checked?c.L*2*sbd+34*sbd:c.L*34*sbd,logit=Bt*T*c.V*4*2;
 const tr=[[4*N,0],[4*N,1],[8*N,2],[act,3],[logit,4]],inf=[[2*N,5],[2*c.L*c.Hkv*r.dh*T*Bt*2,6]];
 const tot=a=>a.reduce((s,x)=>s+x[0],0),mx=Math.max(tot(tr),80*G)*1.06;
 const row=(name,a,lab)=>`<div class="memrow"><span>${name}</span><div class="memstack">${a.map(([v,i])=>`<i style="width:${(100*v/mx).toFixed(2)}%;background:${PARTS[i][1]};box-shadow:0 0 10px ${PARTS[i][1]}55" title="${PARTS[i][0]}: ${bytes(v)}"></i>`).join("")}${[24,40,80].map(g=>`<span class="gpu" style="left:${(100*g*G/mx).toFixed(2)}%">${lab?`<span>${g} GiB</span>`:""}</span>`).join("")}</div><b>${bytes(tot(a))}</b></div>`;
 $("#memFig").innerHTML=row(`Eğitim · ${M.key}`,tr,1)+row(`Çıkarım · ${M.key}`,inf,0)+`<div class="grid3" style="margin-top:6px">${[[bytes(16*N),"durum: ağırlık + gradyan + Adam (16 B/param)"],[bytes(act),`aktivasyon (${ick.checked?"checkpointing açık":"checkpointing kapalı"})`],[bytes(logit),"logit'ler (fp32, ileri + geri)"],[tot(tr)<=80*G?"1 × 80 GiB yeter":`≈ ${Math.ceil(tot(tr)/(80*G))} × 80 GiB`,"tek GPU'ya sığar mı? (FSDP ile bölünür)"]].map(s=>`<div class="stat"><b>${s[0]}</b><span>${s[1]}</span></div>`).join("")}</div>`}
[ib,it,ick].forEach(x=>x.addEventListener("input",render));ick.addEventListener("change",render);onModel(render);render();
}

/* ================= BPE (koklm/bpe.py ile aynı algoritma) ================= */
{
const PAT=/ ?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const CORPUS=("Bir varmış bir yokmuş, evvel zaman içinde kalbur saman içinde, develer tellal iken pireler berber iken. "+
"Bir zamanlar uzak bir köyde yaşlı bir adam varmış. Bu adamın bir oğlu, bir de kızı varmış. "+
"Oğlan her sabah dağa çıkar, kız da evde ekmek yaparmış. Bir gün oğlan dağda bir kuyu bulmuş. "+
"Kuyunun içinden bir ses gelmiş: bana su ver, sana üç dilek vereyim demiş. "+
"Model kelimeleri değil, sayıları görür. Tokenizer metni parçalara böler, her parçaya bir sayı verir. "+
"Sık geçen harf çiftleri birleşir, heceler oluşur, heceler kelimelere dönüşür. ").repeat(3);
const enc=new TextEncoder(),dec=new TextDecoder("utf-8",{fatal:false});
function trainBPE(text,n){const chunks=new Map();for(const p of text.match(PAT)||[])chunks.set(p,(chunks.get(p)||0)+1);
 let seqs=[...chunks].map(([p,f])=>[[...enc.encode(p)],f]);const merges=[];
 for(let k=0;k<n;k++){const cnt=new Map();for(const[ids,f]of seqs)for(let i=0;i<ids.length-1;i++){const key=ids[i]*65536+ids[i+1];cnt.set(key,(cnt.get(key)||0)+f)}
  let best=-1,bc=0;for(const[key,c]of cnt){if(c>bc||(c===bc&&key<best)){bc=c;best=key}}if(bc<2)break;
  const a=Math.floor(best/65536),b=best%65536,nw=256+k;merges.push([a,b]);
  seqs=seqs.map(([ids,f])=>{const o=[];for(let i=0;i<ids.length;i++){if(i<ids.length-1&&ids[i]===a&&ids[i+1]===b){o.push(nw);i++}else o.push(ids[i])}return[o,f]})}
 return merges}
function vocabOf(merges){const v=new Map();for(let i=0;i<256;i++)v.set(i,[i]);merges.forEach(([a,b],i)=>v.set(256+i,[...v.get(a),...v.get(b)]));return v}
function encodeBPE(text,merges,lim){const rank=new Map();merges.slice(0,lim).forEach(([a,b],i)=>rank.set(a*65536+b,i));const out=[];
 for(const p of text.match(PAT)||[]){let ids=[...enc.encode(p)];while(ids.length>1){let bi=-1,br=Infinity;for(let i=0;i<ids.length-1;i++){const r=rank.get(ids[i]*65536+ids[i+1]);if(r!==undefined&&r<br){br=r;bi=i}}if(bi<0)break;
  const a=ids[bi],b=ids[bi+1],o=[];for(let i=0;i<ids.length;i++){if(i<ids.length-1&&ids[i]===a&&ids[i+1]===b){o.push(256+br);i++}else o.push(ids[i])}ids=o}out.push(...ids)}return out}
const MERGES=trainBPE(CORPUS,120),VOC=vocabOf(MERGES);
const sl=$("#i_merge"),inp=$("#bpeIn");sl.max=MERGES.length;
const pal=["#67e8f9","#a5b4fc","#c4b5fd","#f9a8d4","#7dd3fc","#99f6e4","#fbcfe8"];
const show=b=>dec.decode(new Uint8Array(b)).replace(/ /g,"·").replace(/\n/g,"↵");
function render(){const lim=+sl.value;$("#o_merge").textContent=lim;sl.style.setProperty("--p",(100*lim/sl.max)+"%");
 const ids=encodeBPE(inp.value,MERGES,lim),nb=enc.encode(inp.value).length;
 $("#bpeOut").innerHTML=ids.map((id,i)=>{const c=pal[id%pal.length];return`<span style="background:${c};box-shadow:0 0 10px ${c}55;animation-delay:${reduce?0:Math.min(i*6,400)}ms" title="id ${id}">${esc(show(VOC.get(id)))}</span>`}).join("");
 $("#bpeStats").innerHTML=[[fmt(nb),"bayt"],[fmt(ids.length),"token"],[dec2(nb/Math.max(1,ids.length)),"bayt / token"],[fmt(256+lim+1),"sözlük boyutu"]].map(s=>`<div class="stat"><b>${s[0]}</b><span>${s[1]}</span></div>`).join("");
 $("#bpeMerges").innerHTML=MERGES.slice(Math.max(0,lim-8),lim).reverse().map(([a,b],i)=>`<span>${esc(show(VOC.get(a)))} + ${esc(show(VOC.get(b)))} → ${esc(show(VOC.get(256+lim-1-i)))}</span>`).join("")||`<span>henüz birleştirme yok: her bayt bir token</span>`}
sl.addEventListener("input",render);inp.addEventListener("input",render);
let play=0;$("#bpePlay").addEventListener("click",()=>{clearInterval(play);sl.value=0;render();play=setInterval(()=>{if(+sl.value>=+sl.max){clearInterval(play);return}sl.value=+sl.value+2;render()},reduce?1:110)});
render();
}

/* ================= Kod gezgini ================= */
{
const L=window.KOD||[];let cur=L.findIndex(k=>k.id==="att");if(cur<0)cur=0;
function render(){$("#flist").innerHTML=L.map((k,i)=>`<button data-i="${i}" aria-pressed="${i===cur}"><b>${esc(k.baslik)}</b><span>${esc(k.dosya)}:${k.satir}</span></button>`).join("");
 const k=L[cur];if(!k)return;const n=k.kod.split("\n").length;
 $("#codeHead").innerHTML=`<div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;align-items:baseline"><h3 class="gt">${esc(k.baslik)}</h3><a class="small" href="https://github.com/mkor4y/koklm/blob/main/${esc(k.dosya)}#L${k.satir}-L${k.satir+n-1}">${esc(k.dosya)} · satır ${k.satir}–${k.satir+n-1} ↗</a></div><p class="small" style="margin-top:6px">${esc(k.aciklama)}</p>`;
 $("#codeView").innerHTML=codeHTML(k,null)}
$("#flist").addEventListener("click",e=>{const b=e.target.closest("button");if(!b)return;cur=+b.dataset.i;render()});
render();
}

/* ================= Ölçek ================= */
{
const D=[["KökLM küçük",2885888,"bizim"],["KökLM 1B",1235814400,"bizim"],["KökLM 8B",8030261248,"bizim"],["DeepSeek-R1",671e9,"açık, 2025"],["Grok 4.8",2.5e12,"bildirilen"],["Qwen 3.8",3.5e12,"bildirilen"],["Eğitimdeki modeller",1e13,"bildirilen"],["İnsan beyni (sinaps)",1e14,"tahmin"]];
const lo=6,hi=14,word=n=>n>=1e12?(n/1e12).toLocaleString("tr-TR",{maximumFractionDigits:1})+" T":n>=1e9?(n/1e9).toLocaleString("tr-TR",{maximumFractionDigits:2})+" B":(n/1e6).toLocaleString("tr-TR",{maximumFractionDigits:1})+" M";
const box=$("#scale");
box.innerHTML=D.map(([a,n,t])=>{const g=t==="bizim"?`linear-gradient(90deg,${C},${V})`:t==="tahmin"?`linear-gradient(90deg,${V},${P})`:`linear-gradient(90deg,#334155,#64748b)`;return`<div class="bar"><span>${a}<br><span class="tag2">${t}</span></span><span class="tr"><i style="width:0;background:${g};box-shadow:0 0 14px ${t==="bizim"?C:t==="tahmin"?P:"transparent"}" data-w="${((Math.log10(n)-lo)/(hi-lo)*100).toFixed(2)}"></i></span><span class="v">${word(n)}</span></div>`}).join("");
const go=()=>box.querySelectorAll("i[data-w]").forEach((i,k)=>setTimeout(()=>i.style.width=i.dataset.w+"%",reduce?0:k*120));
const io=new IntersectionObserver(es=>{if(es[0].isIntersecting){go();io.disconnect()}},{threshold:.2});io.observe(box);setTimeout(go,9000);
}

/* ================= Testler ve hikâye ================= */
{
const TS=[["Hugging Face Llama ile aynı çıktı","test_hf_llama_ile_ayni_logit","Aynı ağırlıklarla logit farkı < 0,0001; Llama 3.1 RoPE ölçeklemesi dahil."],
["Milyar parametre birebir","test_milyar_parametre_meta_cihazda","1B ve 8B bellek ayırmadan kurulur; 1.235.814.400 ve 8.030.261.248."],
["KV-cache doğru","test_kv_cache_ayni_sonuc","Adım adım üretim, tüm diziyi tek seferde işlemekle aynı sonucu verir."],
["Geleceğe bakmıyor","test_nedensellik","Son token'ı değiştirmek önceki konumların çıktısını değiştirmez."],
["RoPE göreli konum","test_rope_goreli_konum","q_m·k_n skoru yalnızca m − n farkına bağlı."],
["Model öğreniyor","test_bir_egitim_adimi_kaybi_dusurur","Birkaç AdamW adımında kayıp düşer; CI'da 30 adımda 5,79 → 0,93."],
["BPE kayıpsız","test_gidis_donus_ve_sikistirma","Kodla, çöz: metin birebir geri gelir; Türkçe ve emoji dahil."],
["Gerçek ağırlık yükleme","test_hf_klasorden_yukle","safetensors dosyasından yüklenen model, kaynak modelle aynı çıktıyı verir."]];
$("#tests").innerHTML=TS.map((t,i)=>`<div class="glass test" data-r data-d="${i%4}"><b>${t[0]}</b><code>${t[1]}</code><p class="small" style="margin:8px 0 0">${t[2]}</p></div>`).join("");
const SY=[["02:04","Fikir","\"Kendi yapay zekâ modelimizi yapacağız.\" Hedef: büyük modellerin mimarisini matematiğiyle anlamak."],
["02:12","İlk mimari, numpy ile","Llama tarzı ileri geçiş ve 1B / 8B parametre dökümü. Sayılar Meta'nınkiyle ilk denemede tuttu."],
["02:21","PyTorch'a geçiş","GPU'ya hazır model; eğitim, veri hazırlama ve üretim betikleri."],
["02:23","GitHub: mkor4y/koklm","KökLM adı kondu, MIT lisansıyla ilk push."],
["02:30","v0.2.0","Python paketi, gerçek Llama ağırlıklarını yükleme, sıfırdan BPE ve GitHub Actions."],
["02:35","Testler gerçek bir hata yakaladı","Yeni transformers sürümünün RoPE ayar biçimi okunmuyordu; düzeltildi. HF Llama ile aynı logit'ler."],
["02:38","Uçtan uca eğitim CI'da","Tokenizer → veri → eğitim → devam → üretim; kayıp 5,79'dan 0,93'e."],
["02:42","Üretim hatası düzeltildi","Yuvarlanmış sözlüğün fazla token'ları maskelendi."],
["03:07","Tanıtım sitesi","Mimari, matematik, hesaplayıcı ve canlı BPE."],
["03:42","Koyu tema ve 3B yığın","Nöral ağ arka planı, ışık olarak veri akışı."],
["Sabah","Bu sürüm","Blok anatomisi, canlı attention, KV-cache, eğitim döngüsü, GPU belleği ve kod gezgini."]];
$("#story").innerHTML=SY.map((s,i)=>`<li class="glass" data-r data-d="${i%2}"><time>${s[0]}</time><b>${esc(s[1])}</b><span>${esc(s[2])}</span></li>`).join("");
}

/* ================= Kaydırınca beliren sahneler ================= */
{
function countUp(el){if(reduce||el.dataset.done)return;el.dataset.done=1;const N=+el.dataset.count,t0=performance.now();const f=t=>{const k=Math.min(1,(t-t0)/1400);el.textContent=fmt(N*(1-Math.pow(1-k,3)));if(k<1)requestAnimationFrame(f)};requestAnimationFrame(f)}
const rio=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add("in");rio.unobserve(e.target);e.target.querySelectorAll("[data-count]").forEach(countUp)}}),{threshold:.08,rootMargin:"0px 0px -4% 0px"});
document.querySelectorAll("[data-r]").forEach(el=>rio.observe(el));
setTimeout(()=>document.querySelectorAll("[data-r]:not(.in)").forEach(el=>{if(el.getBoundingClientRect().top<innerHeight)el.classList.add("in")}),1500);
const links=[...document.querySelectorAll(".links a")];const nio=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting)links.forEach(a=>a.classList.toggle("on",a.getAttribute("href")==="#"+e.target.id))})},{rootMargin:"-45% 0px -50% 0px"});
links.forEach(a=>{const s=document.querySelector(a.getAttribute("href"));if(s)nio.observe(s)});
}
}
