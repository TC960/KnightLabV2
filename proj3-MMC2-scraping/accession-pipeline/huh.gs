/**
 * MEASUREMENT on BLANK rows (never attempted) that have a pmc_id.
 * Identical extraction/validation/flagging as measurePipeline, but targets
 * the never-touched blanks instead of the pre-failed sentinels.
 * This is the apples-to-apples number that reveals the real recovery rate.
 *
 * Read-only. Writes "_measure_blank".
 */

function measureBlank() {
  const SHEET_NAME  = '';
  const USE_NTH_ACC = 1;
  const SAMPLE      = 100;
  const NCBI_KEY    = '';

  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = SHEET_NAME ? ss.getSheetByName(SHEET_NAME) : ss.getActiveSheet();
  const data = sheet.getDataRange().getValues();
  const headers = data[0].map(h => String(h).trim().toLowerCase());
  const nthCol=(name,n)=>{let s=0;for(let c=0;c<headers.length;c++){if(headers[c]===name){s++;if(s===n)return c;}}return -1;};
  const accCol=nthCol('accession code',USE_NTH_ACC), pmcCol=headers.indexOf('pmc_id');

  // BLANK rows with a pmc_id
  const targets=[];
  for(let i=1;i<data.length;i++){
    if(data[i].every(c=>String(c).trim()==='')) continue;
    const raw=String(data[i][accCol]).trim();
    const pmc=String(data[i][pmcCol]).trim();
    if(raw==='' && pmc) targets.push({row:i+1,pmc:pmc});
  }
  if(!targets.length){ SpreadsheetApp.getUi().alert('no blank rows with pmc_id found'); return; }
  const sample=targets.sort(()=>Math.random()-0.5).slice(0,SAMPLE);

  const normPmc=r=>{let s=r.toUpperCase().replace(/\s+/g,'');if(!s.startsWith('PMC'))s='PMC'+s.replace(/[^0-9]/g,'');return s;};
  const dig=r=>r.replace(/[^0-9]/g,'');
  const key=NCBI_KEY?'&api_key='+NCBI_KEY:'';
  const ncbi=id=>'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id='+id+'&rettype=full&retmode=xml'+key;
  const epmc=p=>'https://www.ebi.ac.uk/europepmc/webservices/rest/'+p+'/fullTextXML';
  const fetchText=p=>{
    try{const r=UrlFetchApp.fetch(ncbi(dig(p)),{muteHttpExceptions:true});
      if(r.getResponseCode()===200){const t=r.getContentText(); if(/<body[\s>]/i.test(t))return t;}}catch(e){}
    Utilities.sleep(200);
    try{const r=UrlFetchApp.fetch(epmc(p),{muteHttpExceptions:true});
      if(r.getResponseCode()===200){const t=r.getContentText(); if(/<body[\s>]/i.test(t))return t;}}catch(e){}
    return '';
  };

  const DICT=[
    ['BioProject','insdc',/PRJ(?:EB|NA|DB|CA)\d{4,}/g],['BioSample','insdc',/SAM(?:EA|N|D)\d{6,}/g],
    ['SRA','insdc',/\b[SED]R[APRSX]\d{6,}/g],['GEO','geo',/\bG(?:SE|SM|PL|DS)\d{4,}/g],
    ['dbGaP','fmt',/\bphs\d{6}(?:\.v\d+\.p\d+)?/g],['GSA','fmt',/\bCR[ARX]\d{4,}/g],
    ['NODE','fmt',/\bOE[PXZS]\d{4,}/g],['CNGB','fmt',/\bCN[PSXR]\d{5,}/g],
    ['MG-RAST','fmt',/\bmg[mp]\d+\.\d+/g],['MetaboLights','fmt',/\bMTBLS\d+/g],
    ['PRIDE','fmt',/\bPXD\d{6}/g],['EGA','fmt',/\bEGA[SD]\d{6,}/g],['ArrayExpress','fmt',/\bE-[A-Z]{4}-\d+/g],
    ['figshare','fmt',/10\.6084\/m9\.figshare\.\d+/g],['Zenodo','fmt',/10\.5281\/zenodo\.\d+/g],['Dryad','fmt',/10\.506\d\/dryad\.\w+/g]
  ];
  const LOOKALIKE=/(refseq|primer|probe|taqman|assay|catalog|cat\.?\s*no|gene (id|expression)|mrna reference|\bNM_|\bNR_|\bXM_|\bNP_)/i;
  const OWN=/(deposit|submitted to|generated in this study|sequences? (were|have been) (deposited|submitted)|available (under|in|at)[^.]{0,60}(accession|bioproject|archive|repositor)|raw (reads|data|sequences)[^.]{0,40}(deposit|available)|under (accession|bioproject))/i;
  const REUSE=/(obtained from|downloaded from|re-?analy|retrieved from|publicly available (data|dataset)[^.]{0,30}(from|under)|previously published|from a previous|reuse|acquired from|accessed from)/i;
  const enaOk=acc=>{try{const r=UrlFetchApp.fetch('https://www.ebi.ac.uk/ena/browser/api/summary/'+acc,{muteHttpExceptions:true});return r.getResponseCode()===200&&/accession/i.test(r.getContentText());}catch(e){return false;}};

  const rows=[]; let recovered=0, floor=0, notext=0, fp=0;
  sample.forEach(s=>{
    const pmc=normPmc(s.pmc); const xml=fetchText(pmc);
    if(!xml){ rows.push([s.row,pmc,'(could not fetch)','','']); notext++; Utilities.sleep(200); return; }
    const plain=xml.replace(/<[^>]+>/g,' ').replace(/\s+/g,' ');
    const found=[]; const seen=new Set();
    DICT.forEach(([repo,kind,re])=>{re.lastIndex=0;let m;while((m=re.exec(plain))){
      const code=m[0].toUpperCase(); if(seen.has(code))continue;
      const at=m.index, win=plain.slice(Math.max(0,at-60),at+code.length+60);
      if(LOOKALIKE.test(win)){fp++;continue;}
      seen.add(code);
      const wide=plain.slice(Math.max(0,at-160),at+code.length+160);
      found.push({code,repo,kind,prov:OWN.test(wide)?'own':(REUSE.test(wide)?'reused':'unclear')});
    }});
    if(!found.length){ floor++; rows.push([s.row,pmc,'N/A','',
      /available (on|upon) request/i.test(plain)?'on request':/no datasets? (were|was)/i.test(plain)?'no data generated':'no accession in text']);
      Utilities.sleep(200); return; }
    found.forEach(f=>{ f.valid = f.kind==='insdc' ? enaOk(f.code) : true; Utilities.sleep(120); });
    const good=found.filter(f=>f.valid), rej=found.filter(f=>!f.valid); fp+=rej.length;
    if(good.length){ recovered++;
      rows.push([s.row,pmc, good.map(f=>f.code+' ('+f.prov+')').join('; '),
        good.map(f=>f.kind==='insdc'?'ENA-ok':'fmt').join('; '),
        rej.length?('dropped: '+rej.map(r=>r.code).join(', ')):'']);
    } else { floor++; rows.push([s.row,pmc,'N/A','','all failed validation: '+rej.map(r=>r.code).join(', ')]); }
    Utilities.sleep(200);
  });

  let d=ss.getSheetByName('_measure_blank'); if(d)d.clear(); else d=ss.insertSheet('_measure_blank');
  d.getRange(1,1,1,5).setValues([['row','pmc','codes (own/reused)','validation','notes']]);
  d.getRange(2,1,rows.length,5).setValues(rows);
  d.setColumnWidths(1,1,60);d.setColumnWidth(2,110);d.setColumnWidth(3,320);d.setColumnWidth(4,120);d.setColumnWidth(5,300);

  const n=rows.length;
  SpreadsheetApp.getUi().alert(
    'BLANK MEASUREMENT ('+n+' never-attempted rows w/ pmc_id) -> "_measure_blank"\n\n'+
    'RECOVERED (validated): '+recovered+'  ('+(100*recovered/n).toFixed(0)+'%)\n'+
    'N/A (floor):           '+floor+'  ('+(100*floor/n).toFixed(0)+'%)\n'+
    'could not fetch:       '+notext+'\n'+
    'false positives dropped:'+fp+'\n\n'+
    'Compare RECOVERED% here vs the sentinel 10%. If much higher, the blank set is\n'+
    'the real prize and the sentinels were genuinely floor.'
  );
}