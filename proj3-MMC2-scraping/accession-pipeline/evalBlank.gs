/**
 * REVERSE EXTRACT — instead of searching for known formats, search for
 * accession CONTEXT and repository URLs, and dump whatever is there.
 * This surfaces accession formats we DON'T yet have in the dictionary
 * (e.g. BioStudies S-BSST836, Qiita, others).
 *
 * For blank+pmc rows that our strict dict finds NOTHING in, it dumps:
 *   - every repository URL in the paper (raw hrefs)
 *   - every "accession number: X" style phrase with the token after it
 * Read-only. Writes "_reverse".
 */

function reverseExtract() {
  const SHEET_NAME='', USE_NTH_ACC=1, SAMPLE=40, NCBI_KEY='';
  const ss=SpreadsheetApp.getActiveSpreadsheet();
  const sheet=SHEET_NAME?ss.getSheetByName(SHEET_NAME):ss.getActiveSheet();
  const data=sheet.getDataRange().getValues();
  const headers=data[0].map(h=>String(h).trim().toLowerCase());
  const nthCol=(name,n)=>{let s=0;for(let c=0;c<headers.length;c++){if(headers[c]===name){s++;if(s===n)return c;}}return -1;};
  const accCol=nthCol('accession code',USE_NTH_ACC), pmcCol=headers.indexOf('pmc_id');

  const targets=[];
  for(let i=1;i<data.length;i++){
    if(data[i].every(c=>String(c).trim()==='')) continue;
    if(String(data[i][accCol]).trim()==='' && String(data[i][pmcCol]).trim())
      targets.push({row:i+1,pmc:String(data[i][pmcCol]).trim()});
  }
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

  // current strict dict (to know if we'd already catch it)
  const STRICT=/PRJ(?:EB|NA|DB|CA)\d{4,}|SAM(?:EA|N|D)\d{6,}|[SED]R[APRSX]\d{6,}|G(?:SE|SM|PL|DS)\d{4,}|phs\d{6}|CR[ARX]\d{4,}|OE[PXZS]\d{4,}|CN[PSXR]\d{5,}|MTBLS\d+|PXD\d{6}|EGA[SD]\d{6,}|E-[A-Z]{4}-\d+|figshare\.\d+|zenodo\.\d+/gi;

  // repository domains -> any URL pointing at these is a data link
  const REPO_URL=/https?:\/\/[^\s"'<>]*(ncbi\.nlm\.nih\.gov|ebi\.ac\.uk|ega-archive|ddbj|cngb\.org|biosino\.org|figshare|zenodo|datadryad|dryad|qiita|mg-rast|metabolights|biostudies|genome\.jp|bioproject|\/sra|\/geo)[^\s"'<>]*/gi;
  // "accession ... TOKEN" — capture the token after accession phrasing, ANY format
  const ACC_PHRASE=/accession(?:\s+(?:number|numbers|no|code|id|ids)?)?[:\s]*([A-Za-z][A-Za-z0-9._-]{4,})/gi;
  const DEPOSIT_PHRASE=/(?:deposited|available|archived|stored)[^.]{0,60}([A-Z][A-Z0-9]{2,}-?[A-Z0-9]*\d{2,})/g;

  const uniq=a=>Array.from(new Set(a));

  const rows=[];
  let hasUnknownUrl=0, hasUnknownPhrase=0, clean=0;
  sample.forEach(s=>{
    const pmc=normPmc(s.pmc); const xml=fetchXml(pmc);
    if(!xml){ rows.push([s.row,pmc,'NO_FETCH','-','-']); Utilities.sleep(200); return; }
    const plain=xml.replace(/<[^>]+>/g,' ').replace(/\s+/g,' ');

    const strictHas=uniq((plain.match(STRICT)||[]).map(x=>x.toUpperCase()));

    // repository URLs (dedupe, trim tracking)
    let urls=uniq((xml.match(REPO_URL)||[]).map(u=>u.replace(/[.,;)]+$/,'')))
      .filter(u=>!/\/(pubmed|pmc|articles|doi\.org|scholar)/i.test(u))   // drop citation links
      .slice(0,5);

    // accession-phrase tokens (dedupe)
    let phrases=[];
    let m; ACC_PHRASE.lastIndex=0;
    while((m=ACC_PHRASE.exec(plain))){ phrases.push(m[1]); }
    m=null; DEPOSIT_PHRASE.lastIndex=0;
    while((m=DEPOSIT_PHRASE.exec(plain))){ phrases.push(m[1]); }
    phrases=uniq(phrases).filter(p=>!/^(number|numbers|code|the|this|are|is|were|and|from|for|available|upon|request|data|these)$/i.test(p)).slice(0,6);

    // is any URL / phrase pointing at something strict WOULDN'T catch?
    const urlNovel = urls.some(u=>!STRICT.test(u));
    STRICT.lastIndex=0;
    const phraseNovel = phrases.some(p=>{STRICT.lastIndex=0; return !STRICT.test(p);});
    if(!strictHas.length && urls.length && urlNovel) hasUnknownUrl++;
    else if(!strictHas.length && phrases.length && phraseNovel) hasUnknownPhrase++;
    else if(!strictHas.length && !urls.length && !phrases.length) clean++;

    rows.push([
      s.row, pmc,
      strictHas.join(', ')||'(none)',
      urls.join('  |  ')||'-',
      phrases.join(', ')||'-'
    ]);
    Utilities.sleep(200);
  });

  let d=ss.getSheetByName('_reverse'); if(d)d.clear(); else d=ss.insertSheet('_reverse');
  d.getRange(1,1,1,5).setValues([['row','pmc','strict_finds','repository_URLs_in_paper','accession_phrase_tokens']]);
  d.getRange(2,1,rows.length,5).setValues(rows);
  d.setColumnWidths(1,1,55);d.setColumnWidth(2,105);d.setColumnWidth(3,150);d.setColumnWidth(4,430);d.setColumnWidth(5,240);

  SpreadsheetApp.getUi().alert(
    'REVERSE EXTRACT ('+rows.length+' blank rows) -> "_reverse"\n\n'+
    'Empty-for-us but has a REPOSITORY URL:      '+hasUnknownUrl+'\n'+
    'Empty-for-us but has an ACCESSION phrase:   '+hasUnknownPhrase+'\n'+
    'Truly clean (no url, no accession phrase):  '+clean+'\n\n'+
    'Read col D (repository URLs) + col E (tokens after "accession"). Anything there\n'+
    'that strict_finds missed = a format we need to add. Send me the odd ones.'
  );
}