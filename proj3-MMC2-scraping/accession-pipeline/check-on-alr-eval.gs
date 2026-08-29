/**
 * BACK-TEST — the measurement with a KNOWN answer.
 * Runs the extractor on 100 rows that ALREADY have a human-entered accession,
 * and checks whether we reproduce the human's code. This tells us if the
 * pipeline actually works, independent of the floor question.
 *
 *   MATCH      = we found the human's code (or a superset containing it)
 *   PARTIAL    = we found some codes but not the human's exact one
 *   MISS       = human has a code, we found nothing
 *   could_not_fetch
 *
 * Read-only. Writes "_backtest".
 */

function backTest() {
  const SHEET_NAME='', USE_NTH_ACC=1, SAMPLE=100, NCBI_KEY='';
  const ss=SpreadsheetApp.getActiveSpreadsheet();
  const sheet=SHEET_NAME?ss.getSheetByName(SHEET_NAME):ss.getActiveSheet();
  const data=sheet.getDataRange().getValues();
  const headers=data[0].map(h=>String(h).trim().toLowerCase());
  const nthCol=(name,n)=>{let s=0;for(let c=0;c<headers.length;c++){if(headers[c]===name){s++;if(s===n)return c;}}return -1;};
  const accCol=nthCol('accession code',USE_NTH_ACC), pmcCol=headers.indexOf('pmc_id');

  // rows with a REAL human code (matches a known format) AND a pmc_id
  const KNOWN=/^(PRJ(EB|NA|DB|CA)\d+|SAM(EA|N|D)\d+|[SED]R[APRSX]\d+|G(SE|SM|PL|DS)\d+|PHS\d+|CR[ARX]\d+|HRA\d+|OE[PXZS]\d+|CN[PSXR]\d+|MG[MP]\d+|MTBLS\d+|PXD\d+|EGA[SD]\d+|E-[A-Z]{4}-\d+)/i;
  const targets=[];
  for(let i=1;i<data.length;i++){
    if(data[i].every(c=>String(c).trim()==='')) continue;
    const raw=String(data[i][accCol]).trim().toUpperCase();
    const pmc=String(data[i][pmcCol]).trim();
    if(pmc && raw && raw!=='N/A' && !/^ACCESSION_NOT_FOUND$/i.test(raw)){
      // pull the human's code tokens that look like real accessions
      const tokens=raw.split(/[\s,;/|]+/).filter(t=>KNOWN.test(t));
      if(tokens.length) targets.push({row:i+1,pmc:pmc,human:tokens.map(t=>t.toUpperCase())});
    }
  }
  if(!targets.length){ SpreadsheetApp.getUi().alert('no human-filled rows with a recognizable code + pmc_id'); return; }
  const sample=targets.sort(()=>Math.random()-0.5).slice(0,SAMPLE);

  const normPmc=r=>{let s=r.toUpperCase().replace(/\s+/g,'');if(!s.startsWith('PMC'))s='PMC'+s.replace(/[^0-9]/g,'');return s;};
  const dig=r=>r.replace(/[^0-9]/g,'');
  const key=NCBI_KEY?'&api_key='+NCBI_KEY:'';
  const ncbi=id=>'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id='+id+'&rettype=full&retmode=xml'+key;
  const epmc=p=>'https://www.ebi.ac.uk/europepmc/webservices/rest/'+p+'/fullTextXML';
  const fetchXml=p=>{
    try{const r=UrlFetchApp.fetch(ncbi(dig(p)),{muteHttpExceptions:true});if(r.getResponseCode()===200){const t=r.getContentText();if(/<body[\s>]/i.test(t))return t;}}catch(e){}
    Utilities.sleep(200);
    try{const r=UrlFetchApp.fetch(epmc(p),{muteHttpExceptions:true});if(r.getResponseCode()===200){const t=r.getContentText();if(/<body[\s>]/i.test(t))return t;}}catch(e){}
    return '';
  };

  // extraction dict — now includes HRA and KAP
  const DICT=/PRJ(?:EB|NA|DB|CA)\d{4,}|SAM(?:EA|N|D)\d{6,}|\b[SED]R[APRSX]\d{6,}|\bG(?:SE|SM|PL|DS)\d{4,}|\bphs\d{6}(?:\.v\d+\.p\d+)?|\bCR[ARX]\d{4,}|\bHRA\d{4,}|\bKAP\d{4,}|\bOE[PXZS]\d{4,}|\bCN[PSXR]\d{5,}|\bmg[mp]\d+\.\d+|\bMTBLS\d+|\bPXD\d{6}|\bEGA[SD]\d{6,}|\bE-[A-Z]{4}-\d+|10\.6084\/m9\.figshare\.\d+|10\.5281\/zenodo\.\d+|10\.506\d\/dryad\.\w+/gi;

  // loose fallback (any digit count) to see if a stricter miss is the cause
  const LOOSE=/PRJ[A-Z]{2}\d+|SAM[A-Z]?\d+|[SED]R[APRSX]\d+|GS[EM]\d+|phs\d+|CR[ARX]\d+|HRA\d+|OE[PX]\d+|CN[PS]\d+|EGA[SD]\d+/gi;

  // normalize: strip version suffixes so PHS000228.V3.P1 matches PHS000228
  const base=c=>c.toUpperCase().replace(/\.V\d+\.P\d+$/,'').replace(/\.\d+$/,'');

  const rows=[]; let match=0, partial=0, miss=0, nofetch=0, looseWouldFix=0;
  sample.forEach(s=>{
    const pmc=normPmc(s.pmc); const xml=fetchXml(pmc);
    if(!xml){ rows.push([s.row,pmc,s.human.join(', '),'(could not fetch)','MISS_FETCH']); nofetch++; Utilities.sleep(200); return; }
    const plain=xml.replace(/<[^>]+>/g,' ').replace(/\s+/g,' ');

    const foundArr=Array.from(new Set((plain.match(DICT)||[]).map(x=>x.toUpperCase())));
    const foundBase=new Set(foundArr.map(base));
    const humanBase=s.human.map(base);
    const gotAll=humanBase.every(h=>foundBase.has(h));
    const gotSome=humanBase.some(h=>foundBase.has(h));

    let status;
    if(gotAll) { status='MATCH'; match++; }
    else if(gotSome){ status='PARTIAL'; partial++; }
    else if(foundArr.length){ status='WRONG_CODES'; partial++; }
    else {
      status='MISS'; miss++;
      // would a looser pattern have caught the human code?
      const loose=Array.from(new Set((plain.match(LOOSE)||[]).map(x=>base(x))));
      if(humanBase.some(h=>loose.indexOf(h)>=0)) { status='MISS(loose-would-fix)'; looseWouldFix++; }
    }
    rows.push([s.row,pmc,s.human.join(', '),foundArr.slice(0,8).join(', ')||'(none)',status]);
    Utilities.sleep(200);
  });

  let d=ss.getSheetByName('_backtest'); if(d)d.clear(); else d=ss.insertSheet('_backtest');
  d.getRange(1,1,1,5).setValues([['row','pmc','human_code','we_found','result']]);
  d.getRange(2,1,rows.length,5).setValues(rows);
  d.setColumnWidths(1,1,55);d.setColumnWidth(2,105);d.setColumnWidth(3,220);d.setColumnWidth(4,260);d.setColumnWidth(5,180);

  const n=rows.length, fetched=n-nofetch;
  const rate=fetched?(100*match/fetched).toFixed(0):'0';
  SpreadsheetApp.getUi().alert(
    'BACK-TEST ('+n+' human-verified rows) -> "_backtest"\n\n'+
    'MATCH (reproduced human code): '+match+'  ('+rate+'% of fetched)\n'+
    'PARTIAL / wrong codes:         '+partial+'\n'+
    'MISS (found nothing):          '+miss+'\n'+
    '  of which loose would fix:    '+looseWouldFix+'\n'+
    'could not fetch:               '+nofetch+'\n\n'+
    'Target is 95%+. If MATCH is high -> pipeline works, blank set is real floor.\n'+
    'If MISS is high -> read "_backtest" col C vs D: those are codes that ARE in\n'+
    'a paper we know has one. That is the real bug to fix.'
  );
}