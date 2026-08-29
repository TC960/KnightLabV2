/*  ACCESSION EXTRACTION — PRODUCTION
 *  Tab: articles.csv   Headers: url | doi | pubmed_id | pmc_id | notes | accession code
 *
 *  CHANGES IN THIS VERSION
 *  1. fetchFullText is now EPMC-FIRST everywhere (not just retry). NCBI is fallback.
 *     Google's Apps Script egress IP is throttled by NCBI; EPMC (EBI, UK) is a
 *     separate server with a separate limit and has never been the thing failing.
 *  2. Your NCBI_KEY was already in CFG and already appended to every efetch call.
 *     So the throttle happened *despite* a key -> the key is likely invalid/rejected.
 *     Run testNcbiKey() to confirm. If it's bad, regenerate at
 *     https://account.ncbi.nlm.nih.gov/settings/  (API Key Management).
 *     With a valid key you get 10 req/s keyed to YOU, not to Google's shared IP.
 *  3. backfillViaPMID no longer needs NCBI idconv. Europe PMC resolves PMID -> PMCID
 *     via the EXT_ID search endpoint. Phase 2 now has ZERO hard NCBI dependency.
 *  4. OWN/REUSE patterns widened substantially (305/329 were coming back "unclear"),
 *     plus a two-stage context window and a paper-level fallback.
 *  5. Circuit breaker + unflagReviews retained. Single fetch path, no duplicate.
 *
 *  NOTE: extraction logic lives here as extractCodes_(). If phase2.gs also defines
 *  extractFromPlain(), leave it — nothing here calls it anymore, so no name clash.
 *
 *  ORDER OF OPERATIONS
 *    testNcbiKey()        -> is the key actually working?
 *    fixRow720()          -> the 1 straggler
 *    unflagReviews()      -> revert the false [review] flags
 *    retryFailures()      -> clear the 705 pending (now EPMC-first)
 *    backfillViaPMID()    -> PMID -> PMCID via EPMC, then extract
 *    backfillViaDOI()     -> DOI -> PMCID via EPMC, then extract
 *    resetExtraction() + installTrigger()  -> full sweep
 */

// ---------- CONFIG ----------
var CFG = {
  SHEET: 'articles.csv',
  NCBI_KEY: '9520fa43f9257d41f8ab9818043ea469bb08', // rotate after the pilot; verify with testNcbiKey()
  LIMIT: 100000,
  PACE_MS: 250,          // EPMC tolerates this fine; it's not the bottleneck
  TIME_BUDGET_MS: 280000
};

var RETRY_MARK  = '[auto] fetch failed - retry';
var REVIEW_MARK = '[review] fetch failed - manual check needed';

// ---------- DICTIONARY ----------
var DICT = [
  ['BioProject',   /PRJ(?:EB|NA|DB|CA)\d{4,}/gi,        true],
  ['BioSample',    /SAM(?:EA|N|D)\d{6,}/gi,             true],
  ['SRA/ENA/DDBJ', /[SED]R[APRSX]\d{6,}/gi,             true],
  ['GEO',          /G(?:SE|SM|PL|DS)\d{4,}/gi,          true],
  ['dbGaP',        /phs\d{6}(?:\.v\d+\.p\d+)?/gi,       true],
  ['GSA',          /CR[ARX]\d{4,}/gi,                   true],
  ['GSA-Human',    /HRA\d{4,}/gi,                       true],
  ['GSA-KAP',      /KAP\d{4,}/gi,                       true],
  ['NODE',         /OE[PXZS]\d{4,}/gi,                  true],
  ['CNGB',         /CN[PSXR]\d{5,}/gi,                  true],
  ['MG-RAST',      /mg[mp]\d+\.\d+/gi,                  true],
  ['MetaboLights', /MTBLS\d+/gi,                        true],
  ['PRIDE',        /PXD\d{6}/gi,                        true],
  ['EGA',          /EGA[SD]\d{6,}/gi,                   true],
  ['ArrayExpress', /E-[A-Z]{4}-\d+/gi,                  true],
  ['BioStudies',   /S-[A-Z]{4}\d+/gi,                   true],
  ['figshare',     /10\.6084\/m9\.figshare\.\d+/gi,     false],
  ['Zenodo',       /10\.5281\/zenodo\.\d+/gi,           false],
  ['Dryad',        /10\.506\d\/dryad\.\w+/gi,           false]
];

var LOOKALIKE = /(refseq|primer|probe|taqman|assay|catalog|cat\.?\s*no|gene (id|expression)|mrna reference|\bNM_|\bNR_|\bXM_|\bNP_)/i;

// ---------- PROVENANCE (widened) ----------
// Previously 305/329 codes fell through to "unclear". The old OWN pattern demanded
// fairly specific phrasings; papers say this a dozen other ways. Broadened below.
var OWN = new RegExp([
  // explicit deposit verbs, any subject
  '(deposit|submitt|uploaded|archived|lodged|stored)',
  // "data availability" boilerplate
  'data (availability|accessibility)',
  // "<thing> are available/can be found/accessed in <repo>"
  '(data|dataset|datasets|reads|raw reads|sequences|sequencing data|raw data|sequence data|genomes?|metagenomes?|reads and metadata|files)\\s*' +
    '(that support|supporting|generated|produced|used)?[^.]{0,60}' +
    '(are|were|is|was|has|have)\\s*(been\\s*)?' +
    '(deposited|submitted|uploaded|made (publicly )?available|available|accessible|archived|released|stored|shared)',
  // "available in/under/at/from/through <repo or accession>"
  'available (in|under|at|from|through|via)[^.]{0,90}' +
    '(accession|bioproject|biosample|sra|ena|geo|ddbj|gsa|cngb|node|ega|arrayexpress|pride|metabolights|zenodo|figshare|dryad|archive|repositor|database)',
  // accession-number framing
  '(under|with|assigned)\\s*(the\\s*)?(accession|bioproject|project|study|submission)\\s*(number|numbers|code|codes|id|ids|no)?',
  '(accession|bioproject|project|study)\\s*(number|numbers|code|codes|id|ids|no)?[^.]{0,25}(is|are|:|=)',
  // "can be found / retrieved from" our repo
  'can be (found|accessed|retrieved|downloaded)\\s*(in|under|at|from|via)',
  // generated-in-this-study
  '(generated|produced|obtained|sequenced)\\s*(in|for|during|as part of)\\s*(this|the (present|current))\\s*(study|work|paper|project)',
  // first-person deposit
  '\\bwe (deposited|submitted|uploaded|have deposited|have submitted|made (the )?data available)',
  // "sequences reported in this paper have been deposited"
  '(reported|described|presented)\\s*(in|by)\\s*this\\s*(paper|study|article)[^.]{0,60}(deposit|available|submitt)'
].join('|'), 'i');

var REUSE = new RegExp([
  '(obtained|downloaded|retrieved|acquired|sourced|collected|taken|derived|extracted)\\s*from[^.]{0,80}' +
    '(sra|ena|geo|ddbj|bioproject|ncbi|ebi|europe pmc|gsa|cngb|node|mg-?rast|qiita|curatedmetagenomicdata|repositor|database|archive|public|published|previous|prior|earlier)',
  're-?analy|re-?used|re-?use\\b|secondary analys|meta-?analys',
  'publicly available (data|dataset|datasets|sequenc|reads|genomes?)',
  'previously (published|deposited|described|reported|generated|released|collected)',
  'from (a |the )?(previous|prior|earlier|published|existing|original)\\s*(study|studies|work|paper|cohort|dataset)',
  '(data|datasets|reads|sequences|samples)\\s*(were|was|are|is)\\s*(originally\\s*)?(published|generated|produced|described)\\s*(by|in)\\s*[^.]{0,40}(et al|\\([12]\\d{3}\\))',
  'et al\\.?\\s*\\([12]\\d{3}\\)[^.]{0,60}(PRJ|SR[APRSX]|GSE|accession)',
  '(existing|third-?party|external|open|archived) (data|dataset|datasets)',
  'accessed (from|via|through|at)\\s*(the\\s*)?(sra|ena|geo|ncbi|ebi|repositor|database)'
].join('|'), 'i');

// ---------- AUTO-RUN ----------
function installTrigger() {
  removeTrigger();
  ScriptApp.newTrigger('runExtraction').timeBased().everyMinutes(10).create();
  SpreadsheetApp.getUi().alert('Auto-run installed: runExtraction fires every 10 min.\n' +
    'Watch the Executions tab. Run removeTrigger() when the sheet is done.');
}
function removeTrigger() {
  ScriptApp.getProjectTriggers().forEach(function (t) {
    if (t.getHandlerFunction() === 'runExtraction') ScriptApp.deleteTrigger(t);
  });
}

// ---------- DIAGNOSTIC: is the NCBI key actually valid? ----------
// The key was already wired into every efetch call, yet you got throttled from
// request #1. That means the key is probably being rejected. This tells you for sure.
function testNcbiKey() {
  var base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=7692476&rettype=full&retmode=xml';
  var withKey = base + '&api_key=' + CFG.NCBI_KEY;

  var out = [];
  [['NO KEY', base], ['WITH KEY', withKey]].forEach(function (pair) {
    try {
      var r = UrlFetchApp.fetch(pair[1], { muteHttpExceptions: true });
      var code = r.getResponseCode();
      var body = r.getContentText().slice(0, 200);
      out.push(pair[0] + ': HTTP ' + code + (code === 200 ? ' OK' : ' -> ' + body));
    } catch (e) {
      out.push(pair[0] + ': THREW ' + e);
    }
  });
  var msg = out.join('\n\n') +
    '\n\nHTTP 200 on both  = NCBI is fine right now, the block lifted.' +
    '\nHTTP 429 on both  = Google egress IP throttled AND the key is not being honored (invalid key).' +
    '\n429 no-key / 200 with-key = key works, and it fixes everything.' +
    '\nHTTP 400 with key = key is malformed/expired -> regenerate it.';
  Logger.log(msg);
  SpreadsheetApp.getUi().alert(msg);
}

// ---------- FETCH: EPMC first, NCBI fallback ----------
function fetchFullText(pmcRaw) {
  var digits = String(pmcRaw).replace(/[^0-9]/g, '');
  if (!digits) return '';
  var pmc = 'PMC' + digits;

  var epmc = 'https://www.ebi.ac.uk/europepmc/webservices/rest/' + pmc + '/fullTextXML';
  var ncbi = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=' +
             digits + '&rettype=full&retmode=xml' + (CFG.NCBI_KEY ? '&api_key=' + CFG.NCBI_KEY : '');

  // primary: Europe PMC (EBI) — not affected by NCBI's block on Google's egress IP
  for (var a = 0; a < 2; a++) {
    try {
      var r = UrlFetchApp.fetch(epmc, { muteHttpExceptions: true });
      if (r.getResponseCode() === 200) {
        var t = r.getContentText();
        if (/<body[\s>]/i.test(t)) return t;
        break; // 200 but no body = EPMC genuinely has no full text; don't retry, fall through
      }
      if (r.getResponseCode() === 404) break; // not in EPMC, go to NCBI
    } catch (e) { Utilities.sleep(400); }
  }

  // fallback: NCBI
  for (var b = 0; b < 2; b++) {
    try {
      var r2 = UrlFetchApp.fetch(ncbi, { muteHttpExceptions: true });
      var c = r2.getResponseCode();
      if (c === 200) { var t2 = r2.getContentText(); if (/<body[\s>]/i.test(t2)) return t2; }
      if (c === 429) Utilities.sleep(1000);
    } catch (e2) { Utilities.sleep(400); }
  }
  return '';
}

// ---------- ID RESOLUTION VIA EPMC (replaces NCBI idconv) ----------
// PMID -> PMCID without touching NCBI. EXT_ID is EPMC's field for PubMed IDs.
function pmidToPmcid(pmid) {
  var id = String(pmid).replace(/[^0-9]/g, '');
  if (!id) return '';
  var url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:' +
            id + '%20AND%20SRC:MED&format=json&resultType=core&pageSize=1';
  return epmcLookupPmcid_(url);
}

// DOI -> PMCID, also via EPMC.
function doiToPmcid(doi) {
  var d = String(doi).trim().replace(/^https?:\/\/(dx\.)?doi\.org\//i, '');
  if (!d) return '';
  var url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%22' +
            encodeURIComponent(d) + '%22&format=json&resultType=core&pageSize=1';
  return epmcLookupPmcid_(url);
}

function epmcLookupPmcid_(url) {
  for (var a = 0; a < 2; a++) {
    try {
      var r = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
      if (r.getResponseCode() !== 200) { Utilities.sleep(400); continue; }
      var j = JSON.parse(r.getContentText());
      var hits = j && j.resultList && j.resultList.result;
      if (hits && hits.length && hits[0].pmcid) return String(hits[0].pmcid); // "PMC7692476"
      return '';
    } catch (e) { Utilities.sleep(400); }
  }
  return '';
}

// ---------- EXTRACTION ----------
function extractCodes_(plain) {
  var seen = {}, codes = [];

  DICT.forEach(function (entry) {
    var repo = entry[0], re = entry[1], guard = entry[2];
    re.lastIndex = 0;
    var m;
    while ((m = re.exec(plain))) {
      var code = m[0].toUpperCase();
      var at = m.index, L = m[0].length;

      if (guard) {
        var before = at > 0 ? plain.charAt(at - 1) : ' ';
        var after = (at + L) < plain.length ? plain.charAt(at + L) : ' ';
        if (/[A-Za-z0-9]/.test(before) || /[A-Za-z]/.test(after)) continue; // not standalone
      }
      if (LOOKALIKE.test(plain.slice(Math.max(0, at - 60), at + L + 60))) continue;

      if (seen[code]) continue;
      seen[code] = true;
      codes.push({ code: code, repo: repo, prov: provenance_(plain, at, L) });
    }
  });

  if (codes.length) return { status: 'OK', codes: codes, reason: '' };
  return { status: 'OK', codes: [], reason: floorReason_(plain) };
}

// Two-stage window + paper-level fallback. This is the main "unclear" fix:
// the old version only looked at +/-160 chars, but Data Availability statements
// routinely put the verb a long way from the accession, or in a prior sentence.
function provenance_(plain, at, L) {
  var near = plain.slice(Math.max(0, at - 200), at + L + 200);
  var own = OWN.test(near), reuse = REUSE.test(near);
  if (own && !reuse) return 'own';
  if (reuse && !own) return 'reused';

  // stage 2: widen
  var wide = plain.slice(Math.max(0, at - 600), at + L + 400);
  own = OWN.test(wide); reuse = REUSE.test(wide);
  if (own && !reuse) return 'own';
  if (reuse && !own) return 'reused';
  if (own && reuse) return 'unclear (mixed)';   // both signals present — genuinely ambiguous

  // stage 3: paper-level. If the paper anywhere says it deposited data and never
  // says it reused any, default to own. This is the common case for a primary study.
  var pOwn = OWN.test(plain), pReuse = REUSE.test(plain);
  if (pOwn && !pReuse) return 'own (paper-level)';
  if (pReuse && !pOwn) return 'reused (paper-level)';
  return 'unclear';
}

function floorReason_(plain) {
  if (/available (on|upon)( the)?( reasonable)? request|from the corresponding author|not publicly available/i.test(plain))
    return 'available on request';
  if (/no datasets? (were|was) (generated|analy[sz]ed)|no new (data|sequenc)/i.test(plain))
    return 'no data generated';
  if (/(included|contained|presented) (with)?in (this )?(published )?(article|paper)|supplementary (information|material|file)|supporting information/i.test(plain))
    return 'data in article/supplement';
  return 'no accession in text';
}

function processPaper(pmcRaw) {
  var xml = fetchFullText(pmcRaw);
  if (!xml) return { status: 'FETCH_FAIL', codes: [], reason: 'fetch failed' };
  var plain = xml.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
  return extractCodes_(plain);
}

// ---------- MAIN ----------
function runExtraction() {
  var t0 = Date.now();
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  if (!sheet) { SpreadsheetApp.getUi().alert('Tab "' + CFG.SHEET + '" not found'); return; }

  var data = sheet.getDataRange().getValues();
  var headers = data[0].map(function (h) { return String(h).trim().toLowerCase(); });
  var pmcCol = headers.indexOf('pmc_id'),
      accCol = headers.indexOf('accession code'),
      notesCol = headers.indexOf('notes');
  if (pmcCol < 0 || accCol < 0 || notesCol < 0) {
    SpreadsheetApp.getUi().alert('Missing a required column.\nHeaders: ' + headers.join(' | '));
    return;
  }

  var props = PropertiesService.getScriptProperties();
  var cursor    = parseInt(props.getProperty('CURSOR') || '1', 10);
  var processed = parseInt(props.getProperty('PROCESSED') || '0', 10);
  var recovered = parseInt(props.getProperty('RECOVERED') || '0', 10);
  var naCount   = parseInt(props.getProperty('NA') || '0', 10);
  var failCount = parseInt(props.getProperty('FAIL') || '0', 10);

  var i = cursor, stoppedForTime = false, consecFails = 0, TRIP = 20;
  var chunkStart = processed;
  Logger.log('=== runExtraction START — cursor %s, %s already processed ===', cursor, processed);

  for (; i < data.length; i++) {
    if (processed >= CFG.LIMIT) break;
    if (Date.now() - t0 > CFG.TIME_BUDGET_MS) { stoppedForTime = true; break; }
    if (consecFails >= TRIP) {
      Logger.log('!!! %s fetches failed in a row — both EPMC and NCBI unreachable. Stopping.', TRIP);
      break;
    }

    var row = data[i];
    if (row.every(function (c) { return String(c).trim() === ''; })) continue;
    if (!isTarget_(row[accCol])) continue;
    var pmc = String(row[pmcCol]).trim();
    if (!pmc) continue; // no PMC -> handled by backfillViaPMID / backfillViaDOI

    var sheetRow = i + 1;
    var res = processPaper(pmc);
    processed++;

    if (res.status === 'FETCH_FAIL') {
      consecFails++;
      sheet.getRange(sheetRow, notesCol + 1).setValue(RETRY_MARK);
      failCount++;
    } else {
      consecFails = 0;
      writeResult_(sheet, sheetRow, accCol, notesCol, res);
      if (res.codes.length) recovered++; else naCount++;
    }

    if (processed % 50 === 0) {
      Logger.log('processed %s | recovered %s · N/A %s · fetch-fail %s | row %s | %ss',
                 processed, recovered, naCount, failCount, sheetRow,
                 Math.round((Date.now() - t0) / 1000));
    }
    Utilities.sleep(CFG.PACE_MS);
  }

  Logger.log('=== chunk end — did %s this run (total %s) ===', processed - chunkStart, processed);
  props.setProperty('CURSOR', String(i));
  props.setProperty('PROCESSED', String(processed));
  props.setProperty('RECOVERED', String(recovered));
  props.setProperty('NA', String(naCount));
  props.setProperty('FAIL', String(failCount));

  var done = (processed >= CFG.LIMIT) || (i >= data.length);
  var triggered = false;
  try { SpreadsheetApp.getUi(); } catch (e) { triggered = true; }
  if (done) { removeTrigger(); Logger.log('=== ALL TARGETS DONE — trigger removed ==='); }
  if (triggered) return;

  SpreadsheetApp.getUi().alert(
    (done ? 'DONE ✅' : 'CHUNK complete — RUN AGAIN to continue') + '\n\n' +
    'processed: ' + processed + '\n' +
    '  recovered (codes): ' + recovered + '\n' +
    '  N/A (floor):       ' + naCount + '\n' +
    '  fetch failed:      ' + failCount + '\n' +
    'next cursor row: ' + (i + 1) +
    (stoppedForTime ? '\n\n(stopped for the 6-min limit — just run again)' : '')
  );
}

// ---------- PASS 2: retry the fetch-failed rows ----------
function retryFailures() {
  var t0 = Date.now();
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var headers = data[0].map(function (h) { return String(h).trim().toLowerCase(); });
  var pmcCol = headers.indexOf('pmc_id'),
      accCol = headers.indexOf('accession code'),
      notesCol = headers.indexOf('notes');

  var fixed = 0, stillFailed = 0, scanned = 0, consecFails = 0;
  var TRIP = 15;
  Logger.log('=== retryFailures START (EPMC-first) ===');

  for (var i = 1; i < data.length; i++) {
    if (Date.now() - t0 > CFG.TIME_BUDGET_MS) break;
    if (consecFails >= TRIP) {
      Logger.log('!!! %s fetches failed in a row — stopping, rows stay retryable.', TRIP);
      break;
    }
    if (String(data[i][notesCol]).trim() !== RETRY_MARK) continue;
    scanned++;
    var sheetRow = i + 1;
    var res = processPaper(String(data[i][pmcCol]).trim());

    if (res.status === 'FETCH_FAIL') {
      consecFails++;
      if (consecFails < TRIP) {
        sheet.getRange(sheetRow, notesCol + 1).setValue(REVIEW_MARK);
        stillFailed++;
      }
    } else {
      consecFails = 0;
      writeResult_(sheet, sheetRow, accCol, notesCol, res);
      fixed++;
    }

    if (scanned % 50 === 0) {
      Logger.log('retried %s | fixed %s · flagged %s | %ss',
                 scanned, fixed, stillFailed, Math.round((Date.now() - t0) / 1000));
    }
    Utilities.sleep(CFG.PACE_MS);
  }
  Logger.log('=== retry chunk end — retried %s, fixed %s, flagged %s ===', scanned, fixed, stillFailed);

  var remaining = 0;
  for (var j = 1; j < data.length; j++) if (String(data[j][notesCol]).trim() === RETRY_MARK) remaining++;
  remaining -= scanned; if (remaining < 0) remaining = 0;

  SpreadsheetApp.getUi().alert(
    (remaining > 0 ? 'CHUNK done — RUN AGAIN to continue' : 'PASS 2 DONE ✅') + '\n\n' +
    'retried this chunk: ' + scanned + '\n' +
    '  fixed on 2nd try: ' + fixed + '\n' +
    '  flagged for human: ' + stillFailed + '\n' +
    'retry rows remaining: ' + remaining
  );
}

// ---------- PHASE 2: rows with no pmc_id ----------
// Resolve PMID -> PMCID (via EPMC, NOT idconv), write it back, then extract.
function backfillViaPMID() { backfill_('pubmed_id', pmidToPmcid, 'PMID'); }
function backfillViaDOI()  { backfill_('doi',       doiToPmcid,  'DOI'); }

function backfill_(sourceColName, resolver, label) {
  var t0 = Date.now();
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var headers = data[0].map(function (h) { return String(h).trim().toLowerCase(); });
  var srcCol   = headers.indexOf(sourceColName),
      pmcCol   = headers.indexOf('pmc_id'),
      accCol   = headers.indexOf('accession code'),
      notesCol = headers.indexOf('notes');
  if (srcCol < 0) { SpreadsheetApp.getUi().alert('Column "' + sourceColName + '" not found'); return; }

  var props = PropertiesService.getScriptProperties();
  var key = 'BACKFILL_' + label;
  var cursor = parseInt(props.getProperty(key) || '1', 10);

  var resolved = 0, unresolved = 0, recovered = 0, naCount = 0, failed = 0, i = cursor;
  Logger.log('=== backfillVia%s START — cursor %s ===', label, cursor);

  for (; i < data.length; i++) {
    if (Date.now() - t0 > CFG.TIME_BUDGET_MS) break;

    var row = data[i];
    if (!isTarget_(row[accCol])) continue;            // already answered
    if (String(row[pmcCol]).trim()) continue;         // already has a PMCID; main pass owns it
    var src = String(row[srcCol]).trim();
    if (!src) continue;

    var sheetRow = i + 1;
    var pmcid = resolver(src);
    Utilities.sleep(CFG.PACE_MS);

    if (!pmcid) {
      unresolved++;
      sheet.getRange(sheetRow, notesCol + 1).setValue('[auto] no PMC full text (' + label + ' unresolved)');
      sheet.getRange(sheetRow, accCol + 1).setValue('N/A');
      continue;
    }

    resolved++;
    sheet.getRange(sheetRow, pmcCol + 1).setValue(pmcid);   // write it back — permanent win

    var res = processPaper(pmcid);
    if (res.status === 'FETCH_FAIL') {
      sheet.getRange(sheetRow, notesCol + 1).setValue(RETRY_MARK);
      failed++;
    } else {
      writeResult_(sheet, sheetRow, accCol, notesCol, res);
      if (res.codes.length) recovered++; else naCount++;
    }

    if (resolved % 25 === 0) {
      Logger.log('%s: resolved %s · unresolved %s · codes %s · N/A %s · fail %s | %ss',
                 label, resolved, unresolved, recovered, naCount, failed,
                 Math.round((Date.now() - t0) / 1000));
    }
    Utilities.sleep(CFG.PACE_MS);
  }

  props.setProperty(key, String(i));
  var done = i >= data.length;
  if (done) props.deleteProperty(key);
  Logger.log('=== backfillVia%s chunk end ===', label);

  SpreadsheetApp.getUi().alert(
    (done ? 'BACKFILL (' + label + ') DONE ✅' : 'CHUNK done — RUN AGAIN to continue') + '\n\n' +
    'resolved to PMCID:  ' + resolved + '\n' +
    'unresolved:         ' + unresolved + '\n' +
    '  codes recovered:  ' + recovered + '\n' +
    '  N/A:              ' + naCount + '\n' +
    '  fetch failed:     ' + failed + '\n' +
    'next cursor row: ' + (i + 1)
  );
}

// ---------- one-offs ----------
function unflagReviews() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var notesCol = data[0].map(function (x) { return String(x).trim().toLowerCase(); }).indexOf('notes');
  var n = 0;
  for (var i = 1; i < data.length; i++) {
    if (String(data[i][notesCol]).trim() === REVIEW_MARK) {
      sheet.getRange(i + 1, notesCol + 1).setValue(RETRY_MARK);
      n++;
    }
  }
  SpreadsheetApp.getUi().alert('Reverted ' + n + ' [review] rows back to retry state.');
}

function fixRow720() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var pmcCol = h.indexOf('pmc_id'), accCol = h.indexOf('accession code'), notesCol = h.indexOf('notes');
  var res = processPaper(String(data[719][pmcCol]).trim());
  if (res.status === 'FETCH_FAIL') {
    SpreadsheetApp.getUi().alert('Row 720 still fetch-failing. Check the PMCID by hand.');
    return;
  }
  writeResult_(sheet, 720, accCol, notesCol, res);
  SpreadsheetApp.getUi().alert('Row 720: ' +
    (res.codes.length ? res.codes.map(function (c) { return c.code; }).join('; ') : 'N/A — ' + res.reason));
}

function resetExtraction() {
  var p = PropertiesService.getScriptProperties();
  ['CURSOR','PROCESSED','RECOVERED','NA','FAIL'].forEach(function (k) { p.deleteProperty(k); });
  SpreadsheetApp.getUi().alert('Cursor reset. Next runExtraction() starts from the top.');
}

// Re-score provenance on rows already coded, without re-fetching anything?
// No — provenance needs the full text, so this DOES re-fetch. Run it only if you
// want the widened OWN/REUSE applied retroactively to the 329 existing codes.
function rescoreProvenance() {
  var t0 = Date.now();
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var pmcCol = h.indexOf('pmc_id'), accCol = h.indexOf('accession code'), notesCol = h.indexOf('notes');

  var props = PropertiesService.getScriptProperties();
  var cursor = parseInt(props.getProperty('RESCORE') || '1', 10);
  var n = 0, i = cursor;

  for (; i < data.length; i++) {
    if (Date.now() - t0 > CFG.TIME_BUDGET_MS) break;
    var note = String(data[i][notesCol]);
    if (note.indexOf('[auto]') !== 0) continue;
    if (note.indexOf('=unclear') < 0) continue;     // only re-score the unclear ones
    var res = processPaper(String(data[i][pmcCol]).trim());
    if (res.status === 'FETCH_FAIL' || !res.codes.length) continue;
    sheet.getRange(i + 1, notesCol + 1).setValue(
      '[auto] ' + res.codes.map(function (c) { return c.code + '=' + c.prov; }).join('; '));
    n++;
    Utilities.sleep(CFG.PACE_MS);
  }
  props.setProperty('RESCORE', String(i));
  if (i >= data.length) props.deleteProperty('RESCORE');
  SpreadsheetApp.getUi().alert('Re-scored ' + n + ' rows. ' +
    (i >= data.length ? 'DONE ✅' : 'Run again to continue.'));
}

// ---------- helpers ----------
function isTarget_(v) {
  var s = String(v).trim();
  return s === '' || /^accession_not_found$/i.test(s);
}

function writeResult_(sheet, sheetRow, accCol, notesCol, res) {
  if (res.codes.length) {
    sheet.getRange(sheetRow, accCol + 1)
         .setValue(res.codes.map(function (c) { return c.code; }).join('; '));
    sheet.getRange(sheetRow, notesCol + 1)
         .setValue('[auto] ' + res.codes.map(function (c) { return c.code + '=' + c.prov; }).join('; '));
  } else {
    sheet.getRange(sheetRow, accCol + 1).setValue('N/A');
    sheet.getRange(sheetRow, notesCol + 1).setValue('[auto] ' + res.reason);
  }
}

function debugRetryRows() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var pmcCol = h.indexOf('pmc_id'), notesCol = h.indexOf('notes');

  var shown = 0;
  for (var i = 1; i < data.length && shown < 5; i++) {
    var note = String(data[i][notesCol]).trim();
    if (note !== RETRY_MARK && note !== REVIEW_MARK) continue;
    shown++;

    var raw = String(data[i][pmcCol]).trim();
    var digits = raw.replace(/[^0-9]/g, '');
    Logger.log('---------- sheet row %s ----------', i + 1);
    Logger.log('raw pmc_id: "%s"  -> digits: "%s"', raw, digits);
    if (!digits) { Logger.log('NO DIGITS — this row has no usable PMCID'); continue; }

    var epmc = 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC' + digits + '/fullTextXML';
    try {
      var r = UrlFetchApp.fetch(epmc, { muteHttpExceptions: true });
      var t = r.getContentText();
      Logger.log('EPMC  HTTP %s | len %s | has <body>: %s',
                 r.getResponseCode(), t.length, /<body[\s>]/i.test(t));
      Logger.log('EPMC  first 300: %s', t.slice(0, 300));
    } catch (e) { Logger.log('EPMC THREW: %s', e); }

    var ncbi = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=' +
               digits + '&rettype=full&retmode=xml';
    try {
      var r2 = UrlFetchApp.fetch(ncbi, { muteHttpExceptions: true });
      var t2 = r2.getContentText();
      Logger.log('NCBI  HTTP %s | len %s | has <body>: %s',
                 r2.getResponseCode(), t2.length, /<body[\s>]/i.test(t2));
      Logger.log('NCBI  first 300: %s', t2.slice(0, 300));
    } catch (e2) { Logger.log('NCBI THREW: %s', e2); }

    Utilities.sleep(400);
  }
  Logger.log('=== inspected %s retry rows ===', shown);
}
