document.addEventListener('DOMContentLoaded', ()=>{
  const file=document.getElementById('plotCsv');
  const type=document.getElementById('chartType');
  const btn=document.getElementById('plotRun');
  const c=document.getElementById('plotCanvas');
  const ctx=c.getContext('2d');

  function parse(text){
    return text.trim().split(/\r?\n/).map(r=>r.split(',').map(Number)).filter(r=>r.length>=2);
  }

  btn.addEventListener('click', async ()=>{
    if(!file.files?.length) return;
    const rows=parse(await file.files[0].text());
    const xs=rows.map(r=>r[0]), ys=rows.map(r=>r[1]);
    const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
    const pad=20, sx=x=>pad+(x-minx)/(maxx-minx||1)*(c.width-2*pad), sy=y=>c.height-(pad+(y-miny)/(maxy-miny||1)*(c.height-2*pad));
    ctx.clearRect(0,0,c.width,c.height);
    ctx.strokeStyle='#111'; ctx.strokeRect(10,10,c.width-20,c.height-20);
    if(type.value==='Scatter'){
      ctx.fillStyle='#111'; rows.forEach(r=>{ctx.beginPath();ctx.arc(sx(r[0]),sy(r[1]),3,0,Math.PI*2);ctx.fill();});
    }else if(type.value==='Line'){
      ctx.beginPath(); rows.forEach((r,i)=>{i?ctx.lineTo(sx(r[0]),sy(r[1])):ctx.moveTo(sx(r[0]),sy(r[1]));}); ctx.stroke();
    }else{
      const bw=(c.width-40)/rows.length; ctx.fillStyle='#111';
      rows.forEach((r,i)=>{const x=20+i*bw; const y=sy(r[1]); ctx.fillRect(x,y,bw*0.8,c.height-20-y);});
    }
  });
});
