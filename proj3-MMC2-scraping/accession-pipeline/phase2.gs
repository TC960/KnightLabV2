/*  PHASE 2 — COMPLETE, SELF-CONTAINED FILE
 *  Replace the entire contents of phase2.gs with this.
 *
 *  THE JOB: 5,712 rows have no accession code and no pmc_id. runExtraction skipped
 *  them all (`if (!pmc) continue`). That's 44% of the corpus sitting empty. This is
 *  the single biggest remaining source of codes. Everything else is polish.
 *
 *  NO PMCID DOES NOT MEAN NO CODES:
 *    Europe PMC has a record for nearly every PubMed article whether or not PMC holds
 *    the full text. That record carries the ABSTRACT. And EBI's text-mined annotations
 *    are keyed to the PMID (MED:xxxx), not the PMCID — the same channel that pulled 78
 *    codes out of papers with zero downloadable text.
 *
 *  PER ROW, CHEAPEST PATH FIRST:
 *    1. One EPMC lookup by DOI (or PMID). Returns pmcid + title + abstract.
 *    2. Check the abstract for codes. FREE — we already have the text. Many rows end here.
 *    3. If PMC has the paper: write the PMCID back, fetch full text, extract.
 *    4. Last resort: EBI mined annotations by PMID.
 *    5. Nothing: N/A with a reason saying which path failed.
 *
 *  CRASH-SAFE: cursor persists every 5 rows, every row is try/caught, 200s budget.
 *
 *  Depends on runExtraction.gs for: CFG, DICT, OWN, REUSE, extractCodes_, isTarget_.
 *  Everything else it needs is in this file.
 */

var P2_BUDGET = 200000;   // 3.3 min — well short of the 6-min Apps Script cap
var P2_SAVE_EVERY = 5;    // persist the cursor this often

// ================= TRIGGERS =================
function installPhase2Trigger() {
  removePhase2Trigger();
  ScriptApp.newTrigger('runPhase2').timeBased().everyMinutes(5).create();
  SpreadsheetApp.getUi().alert(
    'Phase 2 auto-run installed: fires every 5 min.\n\n' +
    'QUOTA HEADS-UP: on a personal Gmail account Apps Script caps TRIGGER runtime at\n' +
    '~90 min/day, so a 5-hour job spreads over several days. A Workspace/university\n' +
    'account gets 6 hr/day and finishes overnight.\n\n' +
    'If you see "Service using too much computer time for one day" — that is quota,\n' +
    'not a bug. The cursor is saved. It resumes where it stopped.');
}

function removePhase2Trigger() {
  ScriptApp.getProjectTriggers().forEach(function (t) {
    if (t.getHandlerFunction() === 'runPhase2') ScriptApp.deleteTrigger(t);
  });
}

// ================= SCOPE CHECK (instant, no network) =================
function phase2Scope() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var doiCol = h.indexOf('doi'), pmidCol = h.indexOf('pubmed_id'),
      pmcCol = h.indexOf('pmc_id'), accCol = h.indexOf('accession code');

  var noPmc = 0, hasPmid = 0, doiOnly = 0, nothing = 0, answered = 0;

  for (var i = 1; i < data.length; i++) {
    var row = data[i];
    if (row.every(function (c) { return String(c).trim() === ''; })) continue;
    if (!isTarget_(row[accCol])) { answered++; continue; }
    if (String(row[pmcCol]).trim()) continue;

    noPmc++;
    var hasP = !!String(row[pmidCol]).replace(/[^0-9]/g, '');
    var hasD = !!String(row[doiCol]).trim();
    if (hasP) hasPmid++;
    else if (hasD) doiOnly++;
    else nothing++;
  }

  var msg = 'PHASE 2 SCOPE\n\n' +
    'already answered: ' + answered + '\n\n' +
    'UNANSWERED, no PMCID: ' + noPmc + '\n' +
    '  has a PMID (best odds): ' + hasPmid + '\n' +
    '  DOI only:               ' + doiOnly + '\n' +
    '  no identifier at all:   ' + nothing;
  Logger.log(msg);
  SpreadsheetApp.getUi().alert(msg);
}

// ================= MAIN =================
function runPhase2() {
  var t0 = Date.now();
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var doiCol   = h.indexOf('doi'),
      pmidCol  = h.indexOf('pubmed_id'),
      pmcCol   = h.indexOf('pmc_id'),
      accCol   = h.indexOf('accession code'),
      notesCol = h.indexOf('notes');

  if (accCol < 0 || notesCol < 0 || pmcCol < 0) {
    SpreadsheetApp.getUi().alert('Missing a required column. Headers: ' + h.join(' | '));
    return;
  }

  // notes and accession code are adjacent -> write both in one call, not two
  var pairFirst = Math.min(notesCol, accCol);
  var adjacent  = Math.abs(accCol - notesCol) === 1;

  var props = PropertiesService.getScriptProperties();
  var cursor = parseInt(props.getProperty('P2_CURSOR') || '1', 10);
  var gotPmc = parseInt(props.getProperty('P2_PMC')   || '0', 10);
  var codes  = parseInt(props.getProperty('P2_CODES') || '0', 10);
  var na     = parseInt(props.getProperty('P2_NA')    || '0', 10);
  var nores  = parseInt(props.getProperty('P2_NORES') || '0', 10);

  var i = cursor, did = 0;
  Logger.log('=== runPhase2 START — cursor %s ===', cursor);

  var save = function (at) {
    props.setProperty('P2_CURSOR', String(at));
    props.setProperty('P2_PMC',   String(gotPmc));
    props.setProperty('P2_CODES', String(codes));
    props.setProperty('P2_NA',    String(na));
    props.setProperty('P2_NORES', String(nores));
  };

  var writeRow = function (sheetRow, code, note) {
    if (adjacent) {
      var vals = (pairFirst === notesCol) ? [[note, code]] : [[code, note]];
      sheet.getRange(sheetRow, pairFirst + 1, 1, 2).setValues(vals);
    } else {
      sheet.getRange(sheetRow, accCol + 1).setValue(code);
      sheet.getRange(sheetRow, notesCol + 1).setValue(note);
    }
  };

  for (; i < data.length; i++) {
    if (Date.now() - t0 > P2_BUDGET) break;

    var row = data[i];
    if (row.every(function (c) { return String(c).trim() === ''; })) continue;
    if (!isTarget_(row[accCol])) continue;              // already answered
    if (String(row[pmcCol]).trim()) continue;           // has a PMCID — runExtraction's job

    var sheetRow = i + 1;

    try {
      var doi  = String(row[doiCol]).trim();
      var pmid = String(row[pmidCol]).replace(/[^0-9]/g, '');

      if (!doi && !pmid) {
        writeRow(sheetRow, 'N/A', '[auto] no identifier (no DOI, no PMID)');
        na++; did++;
        if (did % P2_SAVE_EVERY === 0) save(i + 1);
        continue;
      }

      // ---- call 1: the EPMC record. pmcid + title + abstract, all in one. ----
      var rec = epmcRecord_(doi, pmid);
      Utilities.sleep(150);

      if (!rec) {
        writeRow(sheetRow, 'N/A', '[auto] not found in Europe PMC');
        nores++; did++;
        if (did % P2_SAVE_EVERY === 0) save(i + 1);
        continue;
      }

      if (rec.pmid && !pmid) sheet.getRange(sheetRow, pmidCol + 1).setValue(rec.pmid);

      // ---- free check: is the code in the title/abstract we already have? ----
      var found = [], seen = {}, via = '';
      var absText = ((rec.title || '') + ' ' + (rec.abstract || '')).trim();
      if (absText) {
        extractCodes_(absText).codes.forEach(function (c) {
          if (!seen[c.code]) { seen[c.code] = true; found.push(c); }
        });
        if (found.length) via = 'abstract';
      }

      // ---- call 2, only if needed: full text, when PMC actually has it ----
      if (rec.pmcid) {
        sheet.getRange(sheetRow, pmcCol + 1).setValue(rec.pmcid);   // permanent win
        gotPmc++;
        if (!found.length) {
          var art = fetchArticleP2_(rec.pmcid);
          if (art.kind !== 'FAIL') {
            var plain = art.xml.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
            extractCodes_(plain).codes.forEach(function (c) {
              if (!seen[c.code]) { seen[c.code] = true; found.push(c); }
            });
            if (found.length) via = (art.kind === 'FULL' ? 'full text' : 'front matter');
          }
          Utilities.sleep(150);
        }
      }

      // ---- call 3, last resort: EBI mined annotations, keyed to the PMID ----
      if (!found.length && rec.pmid) {
        var mined = fetchAnnotationsMed_([rec.pmid])[rec.pmid] || [];
        mined.forEach(function (c) {
          if (!seen[c.code]) { seen[c.code] = true; found.push(c); }
        });
        if (found.length) via = 'annotations';
        Utilities.sleep(150);
      }

      if (found.length) {
        writeRow(sheetRow,
                 found.map(function (c) { return c.code; }).join('; '),
                 '[auto] ' + found.map(function (c) { return c.code + '=' + c.prov; }).join('; '));
        codes++;
        Logger.log('  row %s [%s] %s', sheetRow, via,
                   found.map(function (c) { return c.code; }).join('; '));
      } else {
        writeRow(sheetRow, 'N/A', '[auto] ' + (rec.pmcid
          ? 'no accession in text'
          : 'no PMC full text; none in abstract or mined annotations'));
        na++;
      }

    } catch (err) {
      Logger.log('!! row %s threw: %s — skipping', sheetRow, err);
      // row is left untouched, stays a target, gets retried on a later pass
    }

    did++;
    if (did % P2_SAVE_EVERY === 0) save(i + 1);        // crash-safe progress
    if (did % 25 === 0) {
      Logger.log('did %s | pmc %s · codes %s · N/A %s · not-in-epmc %s | %ss',
                 did, gotPmc, codes, na, nores, Math.round((Date.now() - t0) / 1000));
    }
  }

  save(i);
  var done = i >= data.length;
  Logger.log('=== chunk end — %s rows this run | cursor now %s ===', did, i);

  var triggered = false;
  try { SpreadsheetApp.getUi(); } catch (e) { triggered = true; }
  if (done) { removePhase2Trigger(); Logger.log('=== PHASE 2 COMPLETE ==='); }
  if (triggered) return;

  SpreadsheetApp.getUi().alert(
    (done ? 'PHASE 2 DONE ✅' : 'CHUNK done — installPhase2Trigger() and walk away') + '\n\n' +
    '  PMCID recovered: ' + gotPmc + '\n' +
    '  CODES FOUND:     ' + codes + '\n' +
    '  N/A:             ' + na + '\n' +
    '  not in EPMC:     ' + nores + '\n' +
    'cursor: ' + i + ' / ' + data.length
  );
}

// ================= HELPERS (all of them, so this file stands alone) =================

// Resolve a row to a Europe PMC record. DOI first (more reliable), then PMID.
// Returns {pmid, pmcid, title, abstract} or null.
function epmcRecord_(doi, pmid) {
  var queries = [];
  if (doi) {
    var d = String(doi).replace(/^https?:\/\/(dx\.)?doi\.org\//i, '').trim();
    queries.push('DOI:%22' + encodeURIComponent(d) + '%22');
  }
  if (pmid) queries.push('EXT_ID:' + pmid + '%20AND%20SRC:MED');

  for (var q = 0; q < queries.length; q++) {
    var url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=' + queries[q] +
              '&format=json&resultType=core&pageSize=1';
    for (var a = 0; a < 2; a++) {
      try {
        var r = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
        if (r.getResponseCode() !== 200) { Utilities.sleep(500); continue; }
        var res = (JSON.parse(r.getContentText()).resultList || {}).result;
        if (!res || !res.length) break;          // this query found nothing — try the next
        var x = res[0];
        return {
          pmid:     x.pmid  ? String(x.pmid)  : '',
          pmcid:    x.pmcid ? String(x.pmcid) : '',
          title:    x.title || '',
          abstract: x.abstractText || ''
        };
      } catch (e) { Utilities.sleep(400); }
    }
  }
  return null;
}

// Fetch an article by PMCID. EPMC first, NCBI fallback.
// kind: 'FULL' (body present) | 'FRONT' (metadata only, publisher restricted) | 'FAIL'
// Named ...P2_ so it can't collide with fetchArticle in nonOA.gs.
function fetchArticleP2_(pmcRaw) {
  var digits = String(pmcRaw).replace(/[^0-9]/g, '');
  if (!digits) return { kind: 'FAIL', xml: '' };

  var epmc = 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC' + digits + '/fullTextXML';
  var ncbi = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=' +
             digits + '&rettype=full&retmode=xml' + (CFG.NCBI_KEY ? '&api_key=' + CFG.NCBI_KEY : '');

  try {
    var r = UrlFetchApp.fetch(epmc, { muteHttpExceptions: true });
    if (r.getResponseCode() === 200) {
      var t = r.getContentText();
      if (/<body[\s>]/i.test(t)) return { kind: 'FULL', xml: t };
    }
  } catch (e) { Utilities.sleep(300); }

  try {
    var r2 = UrlFetchApp.fetch(ncbi, { muteHttpExceptions: true });
    if (r2.getResponseCode() === 200) {
      var t2 = r2.getContentText();
      if (/<body[\s>]/i.test(t2)) return { kind: 'FULL', xml: t2 };
      if (/<article[\s>]/i.test(t2)) return { kind: 'FRONT', xml: t2 };   // publisher restricted
    }
  } catch (e2) { Utilities.sleep(300); }

  return { kind: 'FAIL', xml: '' };
}

// EBI text-mined accessions, keyed by PMID (MED:) rather than PMCID.
// This is what makes DOI-only rows worth anything — the annotation index is tied to
// the PubMed record, so it works on papers PMC has no copy of at all.
// Returns { "36103583": [{code, repo, prov}, ...], ... }
function fetchAnnotationsMed_(pmids) {
  var out = {};
  if (!pmids || !pmids.length) return out;

  var ids = pmids.map(function (p) {
    return 'MED%3A' + String(p).replace(/[^0-9]/g, '');
  }).join(',');

  var url = 'https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds' +
            '?articleIds=' + ids + '&type=Accession%20Numbers&format=JSON';

  var json = null;
  for (var a = 0; a < 2; a++) {
    try {
      var r = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
      if (r.getResponseCode() !== 200) { Utilities.sleep(600); continue; }
      json = JSON.parse(r.getContentText());
      break;
    } catch (e) { Utilities.sleep(500); }
  }
  if (!json || !json.length) return out;

  json.forEach(function (rec) {
    var pid = String(rec.extId || '');
    if (!pid) return;
    var codes = [], seen = {};

    (rec.annotations || []).forEach(function (ann) {
      var exact = String(ann.exact || '').trim();
      if (!exact) return;

      // Mined strings go through OUR dictionary. That's the safety net: RefSNP
      // rs-numbers and other non-deposit IDs match nothing and drop out for free.
      DICT.forEach(function (entry) {
        var repo = entry[0], re = entry[1];
        re.lastIndex = 0;
        var m = re.exec(exact);
        if (!m || m[0].length !== exact.length) return;   // must be the WHOLE token

        var code = m[0].toUpperCase();
        if (seen[code]) return;
        seen[code] = true;

        var prov;
        if (ann.prefix || ann.postfix) {
          var ctx = String(ann.prefix || '') + ' ' + exact + ' ' + String(ann.postfix || '');
          prov = OWN.test(ctx) ? 'own' : (REUSE.test(ctx) ? 'reused' : 'unclear');
        } else {
          prov = 'unclear (mined, no context)';
        }
        codes.push({ code: code, repo: repo, prov: prov });
      });
    });

    if (codes.length) out[pid] = codes;
  });

  return out;
}

// ================= RESET =================
function resetPhase2() {
  var p = PropertiesService.getScriptProperties();
  ['P2_CURSOR','P2_PMC','P2_CODES','P2_ABS','P2_NA','P2_NORES']
    .forEach(function (k) { p.deleteProperty(k); });
  SpreadsheetApp.getUi().alert('Phase 2 cursor reset. Next runPhase2() starts from the top.');
}

function debugPhase2() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var doiCol = h.indexOf('doi'), pmidCol = h.indexOf('pubmed_id'),
      pmcCol = h.indexOf('pmc_id'), accCol = h.indexOf('accession code'),
      notesCol = h.indexOf('notes');

  var shown = 0;
  for (var i = 1; i < data.length && shown < 5; i++) {
    var row = data[i];
    // the rows phase 2 just marked N/A
    if (!/no PMC full text; none in abstract/i.test(String(row[notesCol]))) continue;
    shown++;

    var doi  = String(row[doiCol]).trim();
    var pmid = String(row[pmidCol]).replace(/[^0-9]/g, '');
    Logger.log('---------- sheet row %s ----------', i + 1);
    Logger.log('doi: "%s" | pmid: "%s" | pmc: "%s"', doi, pmid, String(row[pmcCol]));

    if (doi) {
      var d = doi.replace(/^https?:\/\/(dx\.)?doi\.org\//i, '').trim();
      var u1 = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%22' +
               encodeURIComponent(d) + '%22&format=json&resultType=core&pageSize=1';
      Logger.log('DOI URL: %s', u1);
      try {
        var r1 = UrlFetchApp.fetch(u1, { muteHttpExceptions: true });
        var t1 = r1.getContentText();
        var j1 = JSON.parse(t1);
        var res1 = (j1.resultList || {}).result || [];
        Logger.log('DOI query -> HTTP %s | hitCount %s | results %s',
                   r1.getResponseCode(), j1.hitCount, res1.length);
        if (res1.length) {
          Logger.log('  id=%s pmid=%s pmcid=%s abstractLen=%s title=%s',
                     res1[0].id, res1[0].pmid, res1[0].pmcid,
                     (res1[0].abstractText || '').length,
                     (res1[0].title || '').slice(0, 60));
        } else {
          Logger.log('  RAW: %s', t1.slice(0, 400));
        }
      } catch (e) { Logger.log('DOI query THREW %s', e); }
    }

    if (pmid) {
      var u2 = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:' + pmid +
               '%20AND%20SRC:MED&format=json&resultType=core&pageSize=1';
      try {
        var r2 = UrlFetchApp.fetch(u2, { muteHttpExceptions: true });
        var j2 = JSON.parse(r2.getContentText());
        var res2 = (j2.resultList || {}).result || [];
        Logger.log('PMID query -> HTTP %s | hitCount %s | results %s',
                   r2.getResponseCode(), j2.hitCount, res2.length);
        if (res2.length) {
          Logger.log('  id=%s pmid=%s pmcid=%s abstractLen=%s',
                     res2[0].id, res2[0].pmid, res2[0].pmcid,
                     (res2[0].abstractText || '').length);
        }
      } catch (e) { Logger.log('PMID query THREW %s', e); }

      var u3 = 'https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds' +
               '?articleIds=MED%3A' + pmid + '&type=Accession%20Numbers&format=JSON';
      try {
        var r3 = UrlFetchApp.fetch(u3, { muteHttpExceptions: true });
        Logger.log('annotations -> HTTP %s | %s',
                   r3.getResponseCode(), r3.getContentText().slice(0, 300));
      } catch (e) { Logger.log('annotations THREW %s', e); }
    }
    Utilities.sleep(400);
  }
  Logger.log('=== inspected %s rows ===', shown);
}

