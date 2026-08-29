/*  PROVENANCE, ATTEMPT 3 — PRIORITY BY DATE  (+ fixes for attempt 2)
 *
 *  WHAT THE LAST PROBE ESTABLISHED
 *    1. esearch was fine. Every query form resolved the UID. My field tag was not the bug.
 *    2. bioproject -> pubmed elink is EMPTY. Not "links to other papers" — no linksetdbs
 *       at all. NCBI only fills that table when a submitter registers a publication, and
 *       most never do. The BioProject registry channel is DEAD. Real finding, not a bug.
 *    3. GEO works: dbfrom=gds returns linkname "gds_pubmed" with a real PMID.
 *    4. We are rate-limited at 3/sec because the NCBI key is invalid (the 429 body echoes
 *       an IPv6 address in the api-key field). My resolver fired esearch+elink with no
 *       gap, so ~half the elinks 429'd and were silently miscounted as "no link".
 *       => the "45 no registry link" number from that run is CONTAMINATED. Ignore it.
 *
 *  THE NEW IDEA — PRIORITY BY DATE
 *    EPMC indexes which articles mention an accession (same machinery that mined the
 *    codes). So: ask EPMC for every article mentioning PRJNA656590, sort by publication
 *    date. If OUR paper is the earliest mentioner, the authors almost certainly deposited
 *    it -> own. If an earlier paper already mentioned it, ours is citing it -> reused.
 *
 *    Why this is worth trying: it needs only a MENTION, not a registry link. So it reaches
 *    SRP / SAMN / ERP / PHS — the 374 codes the elink route can never touch.
 *
 *  WHY IT MIGHT NOT WORK — read this before trusting a single number
 *    - EPMC's mention index is built mostly from OA full text + abstracts. An earlier
 *      NON-OA paper that mentioned the accession may be invisible. That biases the method
 *      toward false "own". This is the main threat and the probe is designed to expose it.
 *    - Same-month publications are ties. Deposit-then-publish gaps make dates noisy.
 *    - So: DO NOT APPLY THIS TO THE SHEET YET. probeEpmcPriority() writes NOTHING.
 *      It scores the method against rows where the full-text regex already gave a
 *      confident own/reused, and prints an agreement matrix. If agreement is poor,
 *      we bin the idea and report those codes as 'unclear' with a one-line reason.
 *      That is a perfectly defensible result for Sam.
 *
 *  Depends on runExtraction.gs for CFG.
 */

// ================= PACED EUTILS (fixes the 429 miscount) =================
var LAST_EUTILS = 0;
var EUTILS_GAP  = 400;   // ms between ANY two eutils calls. 3/sec shared-IP limit.

function eutilsFetch_(url) {
  for (var attempt = 0; attempt < 3; attempt++) {
    var wait = EUTILS_GAP - (Date.now() - LAST_EUTILS);
    if (wait > 0) Utilities.sleep(wait);
    LAST_EUTILS = Date.now();

    try {
      var r = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
      var c = r.getResponseCode();
      if (c === 200) return r.getContentText();
      if (c === 429) { Utilities.sleep(1200 * (attempt + 1)); continue; }  // back off, retry
      return '';                                                            // other error, give up
    } catch (e) { Utilities.sleep(600); }
  }
  return '';   // exhausted retries — caller must treat this as UNKNOWN, not as "no link"
}

// ================= GEO-ONLY REGISTRY RESOLVER (the one channel that works) =================
// BioProject is dead (no elink table). GEO is live. This does GSE* only, honestly.
function resolveGeoProvenance() {
  var t0 = Date.now();
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var pmidCol = h.indexOf('pubmed_id'), notesCol = h.indexOf('notes');

  var props = PropertiesService.getScriptProperties();
  var cursor = parseInt(props.getProperty('GEOPROV') || '1', 10);
  var toOwn = 0, toReused = 0, noLink = 0, failed = 0, i = cursor;

  for (; i < data.length; i++) {
    if (Date.now() - t0 > CFG.TIME_BUDGET_MS) break;
    var note = String(data[i][notesCol]);
    if (note.indexOf('[auto]') !== 0 || note.indexOf('unclear') < 0) continue;
    if (!/GSE\d+=unclear/i.test(note)) continue;         // GEO codes only

    var myPmid = String(data[i][pmidCol]).replace(/[^0-9]/g, '');
    var changed = false;

    var rebuilt = note.replace(/^\[auto\]\s*/, '').split(';').map(function (p) {
      var seg = p.trim(); if (!seg) return '';
      var eq = seg.lastIndexOf('='); if (eq < 0) return seg;
      var code = seg.slice(0, eq).trim().toUpperCase();
      var prov = seg.slice(eq + 1).trim();
      if (prov.indexOf('unclear') !== 0) return seg;
      if (!/^GSE\d+$/i.test(code)) return seg;

      var uid = geoUid_(code);
      if (!uid) { failed++; return seg; }
      var pmids = geoLinkedPmids_(uid);
      if (pmids === null) { failed++; return seg; }      // fetch failed — NOT "no link"
      if (!pmids.length) { noLink++; return seg; }

      changed = true;
      if (myPmid && pmids.indexOf(myPmid) >= 0) { toOwn++; return code + '=own (registry)'; }
      toReused++; return code + '=reused (registry)';
    }).filter(function (s) { return s; });

    if (changed) sheet.getRange(i + 1, notesCol + 1).setValue('[auto] ' + rebuilt.join('; '));
  }

  props.setProperty('GEOPROV', String(i));
  var done = i >= data.length;
  if (done) props.deleteProperty('GEOPROV');

  SpreadsheetApp.getUi().alert(
    (done ? 'GEO PROVENANCE DONE ✅' : 'CHUNK done — RUN AGAIN') + '\n\n' +
    'own (registry):    ' + toOwn + '\n' +
    'reused (registry): ' + toReused + '\n' +
    'genuinely no link: ' + noLink + '\n' +
    'lookup failed (left alone, retryable): ' + failed
  );
}

function geoUid_(gse) {
  var t = eutilsFetch_('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&term=' +
                       encodeURIComponent(gse + '[GEO Accession]') + '&retmode=json' +
                       (CFG.NCBI_KEY ? '&api_key=' + CFG.NCBI_KEY : ''));
  if (!t) return '';
  try {
    var ids = JSON.parse(t).esearchresult.idlist || [];
    // the series UID is the one prefixed 200... — pick it, not the samples
    for (var i = 0; i < ids.length; i++) if (String(ids[i]).indexOf('200') === 0) return ids[i];
    return ids.length ? ids[0] : '';
  } catch (e) { return ''; }
}

// returns array of PMIDs, or NULL if the fetch failed (caller must not read null as "empty")
function geoLinkedPmids_(uid) {
  var t = eutilsFetch_('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=gds' +
                       '&db=pubmed&id=' + uid + '&retmode=json' +
                       (CFG.NCBI_KEY ? '&api_key=' + CFG.NCBI_KEY : ''));
  if (!t) return null;
  try {
    var out = [];
    (JSON.parse(t).linksets || []).forEach(function (ls) {
      (ls.linksetdbs || []).forEach(function (db) {
        if (db.dbto === 'pubmed') (db.links || []).forEach(function (id) { out.push(String(id)); });
      });
    });
    return out;
  } catch (e) { return null; }
}

// ================= EPMC PRIORITY — THE PROBE. WRITES NOTHING. =================
// Scores "earliest mentioner == depositor" against rows where the full-text regex
// already produced a confident own/reused. Prints an agreement matrix.
function probeEpmcPriority() {
  var t0 = Date.now();
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var pmidCol = h.indexOf('pubmed_id'), notesCol = h.indexOf('notes');

  var SAMPLE = 25;   // per class. keep it small, this is a validation run.
  var cases = [];

  for (var i = 1; i < data.length && cases.length < SAMPLE * 2; i++) {
    var note = String(data[i][notesCol]);
    if (note.indexOf('[auto]') !== 0) continue;
    var pmid = String(data[i][pmidCol]).replace(/[^0-9]/g, '');
    if (!pmid) continue;

    // only take codes the FULL-TEXT regex labelled confidently (not paper-level, not mined)
    var m = /([A-Z0-9._\/]+)=(own|reused)(?:;|$)/i.exec(note);
    if (!m) continue;
    var label = m[2].toLowerCase();
    var nOwn = cases.filter(function (c) { return c.label === 'own'; }).length;
    var nRe  = cases.filter(function (c) { return c.label === 'reused'; }).length;
    if (label === 'own' && nOwn >= SAMPLE) continue;
    if (label === 'reused' && nRe >= SAMPLE) continue;

    cases.push({ row: i + 1, pmid: pmid, code: m[1].toUpperCase(), label: label });
  }

  Logger.log('=== probeEpmcPriority — %s validation cases ===', cases.length);

  var matrix = { own: { own: 0, reused: 0, unknown: 0 }, reused: { own: 0, reused: 0, unknown: 0 } };

  cases.forEach(function (c) {
    if (Date.now() - t0 > CFG.TIME_BUDGET_MS) return;

    var verdict = priorityVerdict_(c.code, c.pmid);
    matrix[c.label][verdict.call]++;
    Logger.log('row %s | %s | regex=%s | priority=%s | ourDate=%s earliestOther=%s (%s mentioners)',
               c.row, c.code, c.label, verdict.call, verdict.ourDate || '?',
               verdict.earliestOther || '-', verdict.n);
    Utilities.sleep(250);
  });

  var agree = matrix.own.own + matrix.reused.reused;
  var judged = agree + matrix.own.reused + matrix.reused.own;
  var pct = judged ? Math.round(100 * agree / judged) : 0;

  var msg =
    'PRIORITY-BY-DATE PROBE (nothing written)\n\n' +
    'regex said OWN     -> priority own ' + matrix.own.own +
      ' | reused ' + matrix.own.reused + ' | unknown ' + matrix.own.unknown + '\n' +
    'regex said REUSED  -> priority own ' + matrix.reused.own +
      ' | reused ' + matrix.reused.reused + ' | unknown ' + matrix.reused.unknown + '\n\n' +
    'agreement where both judged: ' + pct + '%  (' + agree + '/' + judged + ')\n\n' +
    'Read it like this:\n' +
    '  >85% and reused->own is LOW  = method is sound, apply it.\n' +
    '  many regex-REUSED called own = the OA-only mention index is hiding earlier\n' +
    '    papers, exactly the bias we feared. Bin the method.\n' +
    '  lots of unknown = EPMC does not index these accessions. Bin it.';
  Logger.log(msg);
  SpreadsheetApp.getUi().alert(msg);
}

// For one accession: who mentions it, and are we the earliest?
function priorityVerdict_(code, myPmid) {
  var hits = epmcMentioners_(code);
  if (hits === null) return { call: 'unknown', n: 0 };
  if (!hits.length)  return { call: 'unknown', n: 0 };

  var ourDate = null, others = [];
  hits.forEach(function (hh) {
    if (hh.pmid && hh.pmid === myPmid) { ourDate = hh.date; }
    else if (hh.date) others.push(hh.date);
  });

  // our own paper may not be indexed as a mentioner (non-OA). fall back to its pub date.
  if (!ourDate) ourDate = epmcDateForPmid_(myPmid);
  if (!ourDate) return { call: 'unknown', n: hits.length };

  if (!others.length) return { call: 'own', n: hits.length, ourDate: ourDate };

  others.sort();
  var earliest = others[0];

  // same-month = tie = treat as own (deposit-and-publish, companion papers)
  var call = (ourDate.slice(0, 7) <= earliest.slice(0, 7)) ? 'own' : 'reused';
  return { call: call, n: hits.length, ourDate: ourDate, earliestOther: earliest };
}

// every article EPMC knows mentions this accession, with publication dates.
// returns null on fetch failure (distinct from an empty list).
function epmcMentioners_(code) {
  var url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22' +
            encodeURIComponent(code) + '%22&format=json&pageSize=100&resultType=lite';
  for (var a = 0; a < 2; a++) {
    try {
      var r = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
      if (r.getResponseCode() !== 200) { Utilities.sleep(500); continue; }
      var j = JSON.parse(r.getContentText());
      var res = (j.resultList && j.resultList.result) || [];
      return res.map(function (x) {
        return {
          pmid: x.pmid ? String(x.pmid) : '',
          date: x.firstPublicationDate || (x.pubYear ? x.pubYear + '-01-01' : '')
        };
      }).filter(function (x) { return x.date; });
    } catch (e) { Utilities.sleep(400); }
  }
  return null;
}

function epmcDateForPmid_(pmid) {
  var url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:' + pmid +
            '%20AND%20SRC:MED&format=json&pageSize=1&resultType=lite';
  try {
    var r = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
    if (r.getResponseCode() !== 200) return '';
    var res = JSON.parse(r.getContentText()).resultList.result;
    if (!res || !res.length) return '';
    return res[0].firstPublicationDate || (res[0].pubYear ? res[0].pubYear + '-01-01' : '');
  } catch (e) { return ''; }
}