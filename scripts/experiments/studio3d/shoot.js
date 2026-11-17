const {chromium}=require('/opt/npm-tools/node_modules/playwright');const fs=require('fs'),path=require('path');
const style=process.argv[2],only=process.argv[3];
(async()=>{const b=await chromium.launch({args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
const pg=await b.newPage({viewport:{width:720,height:800},deviceScaleFactor:1});
pg.on('console',m=>{if(m.type()==='error')console.log('ERR',m.text())});pg.on('pageerror',e=>console.log('PAGEERR',e.message));
await pg.route('http://local/**',r=>{let u=new URL(r.request().url()).pathname;let f=u.startsWith('/three/')?path.join(__dirname,'node_modules',u):path.join(__dirname,u);
 const ct=f.endsWith('.js')?'text/javascript':f.endsWith('.html')?'text/html':f.endsWith('.json')?'application/json':'font/ttf';
 fs.existsSync(f)?r.fulfill({body:fs.readFileSync(f),contentType:ct}):r.fulfill({status:404,body:''})});
await pg.goto('http://local/scene.html?style='+style);await pg.waitForFunction('window.READY===true',{timeout:60000});await pg.evaluate(()=>document.fonts.ready);
const n=await pg.evaluate('NF');const dir='fr_'+style;fs.mkdirSync(dir,{recursive:true});
const idx=only?only.split(',').map(Number):[...Array(n).keys()];
for(const i of idx){await pg.evaluate(i=>renderFrame(i),i);await pg.screenshot({path:`${dir}/${String(i).padStart(4,'0')}.png`});}
await b.close();console.log('done',idx.length)})();
