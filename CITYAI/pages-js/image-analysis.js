document.addEventListener('DOMContentLoaded', ()=>{
  const file=document.getElementById('imgFile');
  const btn=document.getElementById('imgRun');
  const cv=document.getElementById('imgCanvas');
  const ctx=cv.getContext('2d');

  btn.addEventListener('click', async ()=>{
    if(!file.files?.length) return;
    const img=new Image();
    img.src=URL.createObjectURL(file.files[0]);
    await img.decode();
    const scale=Math.min(cv.width/img.width, cv.height/img.height);
    const w=img.width*scale, h=img.height*scale;
    ctx.clearRect(0,0,cv.width,cv.height);
    ctx.drawImage(img, (cv.width-w)/2, (cv.height-h)/2, w, h);
    const data=ctx.getImageData(0,0,cv.width,cv.height);
    for(let i=0;i<data.data.length;i+=4){
      const y=0.2126*data.data[i]+0.7152*data.data[i+1]+0.0722*data.data[i+2];
      data.data[i]=data.data[i+1]=data.data[i+2]=y;
    }
    ctx.putImageData(data,0,0);
  });
});
