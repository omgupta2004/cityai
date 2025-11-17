// Lightweight PCA using covariance + eigendecomposition via power method (demo)
document.addEventListener('DOMContentLoaded', () => {
  const file = document.getElementById('pcaCsv');
  const compEl = document.getElementById('comp');
  const btn = document.getElementById('pcaRun');
  const canvas = document.getElementById('pcaPlot');
  const ctx = canvas.getContext('2d');

  function parse(text){
    return text.trim().split(/\r?\n/).map(r=>r.split(',').map(Number)).filter(r=>r.every(n=>!Number.isNaN(n)));
  }
  function standardize(X){
    const m = X.length, n = X[0].length;
    const means = Array(n).fill(0);
    for(const r of X) for(let j=0;j<n;j++) means[j]+=r[j];
    for(let j=0;j<n;j++) means[j]/=m;
    const Z = X.map(r=>r.map((v,j)=>v-means[j]));
    return Z;
  }
  function cov(Z){
    const m=Z.length,n=Z[0].length; const C=Array.from({length:n},()=>Array(n).fill(0));
    for(let i=0;i<n;i++) for(let j=i;j<n;j++){
      let s=0; for(const r of Z) s+=r[i]*r[j];
      C[i][j]=C[j][i]=s/(m-1);
    }
    return C;
  }
  function matVec(A,v){const n=A.length;const r=Array(n).fill(0);for(let i=0;i<n;i++)for(let j=0;j<n;j++)r[i]+=A[i][j]*v[j];return r;}
  function norm(v){return Math.hypot(...v)||1;}
  function powerIter(A,steps=100){
    let v=Array(A.length).fill(0).map(()=>Math.random());
    for(let s=0;s<steps;s++){v=matVec(A,v);const nv=norm(v);v=v.map(x=>x/nv);}
    const Av=matVec(A,v); const lambda=Av.reduce((a,b,i)=>a+b*v[i],0);
    return {vec:v, val:lambda};
  }
  function deflate(A,vec,val){
    const n=A.length; const B=Array.from({length:n},(_,i)=>A[i].slice());
    for(let i=0;i<n;i++) for(let j=0;j<n;j++) B[i][j]-=val*vec[i]*vec[j];
    return B;
  }

  function project(Z, comps=2){
    let C = cov(Z);
    const eigs=[];
    for(let k=0;k<comps;k++){
      const {vec,val}=powerIter(C,60);
      eigs.push(vec);
      C = deflate(C,vec,val);
    }
    const E = eigs; // n x comps
    // Z (m x n) * E (n x comps) -> Y (m x comps)
    const m=Z.length,n=Z[0].length;
    const Y=Array.from({length:m},()=>Array(E.length).fill(0));
    for(let i=0;i<m;i++) for(let k=0;k<E.length;k++) for(let j=0;j<n;j++) Y[i][k]+=Z[i][j]*E[k][j];
    return Y;
  }

  function draw(Y){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    const xs=Y.map(p=>p[0]), ys=Y.map(p=>p[1]);
    const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
    const pad=20;
    const sx=x=>pad+(x-minx)/(maxx-minx||1)*(canvas.width-2*pad);
    const sy=y=>canvas.height-(pad+(y-miny)/(maxy-miny||1)*(canvas.height-2*pad));
    Y.forEach(p=>{ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),3,0,Math.PI*2);ctx.fill();});
  }

  btn.addEventListener('click', async ()=>{
    if(!file.files?.length) return;
    const data = parse(await file.files[0].text());
    const Z = standardize(data);
    const comps = Math.max(2, Math.min(3, parseInt(compEl.value||'2',10)));
    const Y = project(Z, comps);
    draw(Y);
  });
});
