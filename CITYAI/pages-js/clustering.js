// Demo k-means (tiny, not for production)
document.addEventListener('DOMContentLoaded', () => {
  const file = document.getElementById('clCsv');
  const kEl = document.getElementById('kInput');
  const btn = document.getElementById('clRun');
  const canvas = document.getElementById('clPlot');
  const ctx = canvas.getContext('2d');

  function parseCSV(text){
    const rows = text.trim().split(/\r?\n/).map(r => r.split(',').map(Number));
    return rows.filter(r => r.length >= 2 && r.every(n => !Number.isNaN(n))).map(r => r.slice(0,2));
  }

  function kmeans(points, k=3, iters=15){
    const centroids = points.slice(0,k).map(p => p.slice());
    for (let t=0; t<iters; t++){
      const groups = Array.from({length:k}, () => []);
      for (const p of points){
        let best=0, bestd=Infinity;
        for (let i=0;i<k;i++){
          const d=(p[0]-centroids[i][0])**2+(p[1]-centroids[i][1])**2;
          if(d<bestd){bestd=d;best=i;}
        }
        groups[best].push(p);
      }
      for (let i=0;i<k;i++){
        if(groups[i].length){
          const sx=groups[i].reduce((a,b)=>a+b[0],0);
          const sy=groups[i].reduce((a,b)=>a+b[1],0);
          centroids[i]=[sx/groups[i].length, sy/groups[i].length];
        }
      }
    }
    const labels = points.map(p=>{
      let best=0,bestd=Infinity;
      for (let i=0;i<k;i++){
        const d=(p[0]-centroids[i][0])**2+(p[1]-centroids[i][1])**2;
        if(d<bestd){bestd=d;best=i;}
      }
      return best;
    });
    return {centroids,labels};
  }

  function draw(points, labels, k){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    const colors = ['#000','#666','#aaa','#333','#999','#ccc'];
    const xs=points.map(p=>p[0]), ys=points.map(p=>p[1]);
    const minx=Math.min(...xs), maxx=Math.max(...xs);
    const miny=Math.min(...ys), maxy=Math.max(...ys);
    const pad=20;
    function sx(x){return pad+(x-minx)/(maxx-minx||1)*(canvas.width-2*pad)}
    function sy(y){return canvas.height-(pad+(y-miny)/(maxy-miny||1)*(canvas.height-2*pad))}
    points.forEach((p,i)=>{
      ctx.fillStyle = colors[labels[i]%colors.length];
      ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 3, 0, Math.PI*2); ctx.fill();
    });
  }

  btn.addEventListener('click', async ()=>{
    if(!file.files?.length) return;
    const text = await file.files[0].text();
    const pts = parseCSV(text);
    const k = Math.max(2, Math.min(20, parseInt(kEl.value||'3',10)));
    const {labels} = kmeans(pts, k, 20);
    draw(pts, labels, k);
  });
});
