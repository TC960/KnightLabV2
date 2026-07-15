/*  FLAGGING — answer every row, mark what needs human eyes
 *
 *  PRINCIPLE (Sam's, and the right one): we are not trying to be deterministic.
 *  Every row gets an answer. Any row with ambiguity gets FLAGGED so a curator can
 *  target it. Ambiguity is signal, not failure. A flagged row is a working row.
 *
 *  Adds a "flag" column (created if absent) with a short machine-readable token,
 *  and a "flag_detail" column with the human-readable why. Notes are left alone.
 *
 *  TWO PASSES
 *    flagStatic()              — no network. Classifies every row from what's already
 *                                in the sheet. Seconds to run. Do this first.
 *    flagProvenanceConflicts() — network. Cross-checks each coded row's own/reused
 *                                label against EPMC priority-by-date. Where the two
 *                                disagree, flags it. Resumable.
 *    flagSummary()             — counts per flag. This is the scorecard for Sam.
 *
 *  On the priority cross-check: as an AUTO-LABELER it was a coin flip (5 agree,
 *  5 disagree on the only cases with signal), so we are NOT using it to write
 *  provenance. We are using it as a SECOND OPINION. Two independent methods
 *  disagreeing is precisely the definition of a row worth a human's time.
 *
 *  Depends on runExtraction.gs for CFG.
 */

var FLAGS = {
  CLEAN:            'CLEAN',              // coded, provenance confident, no conflict
  PROV_UNCLEAR:     'PROV_UNCLEAR',       // coded, but own/reused could not be determined
  PROV_CONFLICT:    'PROV_CONFLICT',      // regex and priority-by-date disagree — HIGH VALUE
  MINED_NO_CONTEXT: 'MINED_NO_CONTEXT',   // code from EBI text-mining, never saw the text
  NO_FULLTEXT:      'NO_FULLTEXT',        // publisher restricted, nothing to read
  SUPPLEMENT:       'SUPPLEMENT',         // "data in article/supplement" — RECOVERABLE later
  ON_REQUEST:       'ON_REQUEST',         // true floor
  NO_DATA:          'NO_DATA',            // true floor
  NO_ACCESSION:     'NO_ACCESSION',       // full text read, genuinely no code in it
  MULTI_CODE:       'MULTI_CODE',         // 4+ codes — curator may need to pick the primary
  PENDING:          'PENDING'             // still unresolved
};

// ---------- schema ----------
function ensureFlagColumns_(sheet) {
  var head = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0]
                  .map(function (x) { return String(x).trim().toLowerCase(); });
  var flagCol = head.indexOf('flag');
  var detailCol = head.indexOf('flag_detail');

  if (flagCol < 0) {
    flagCol = sheet.getLastColumn();
    sheet.getRange(1, flagCol + 1).setValue('flag');
  }
  if (detailCol < 0) {
    detailCol = sheet.getLastColumn();
    sheet.getRange(1, detailCol + 1).setValue('flag_detail');
  }
  return { flag: flagCol, detail: detailCol };
}

// ================= PASS 1: STATIC. No network. Run this first. =================
function flagStatic() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var cols = ensureFlagColumns_(sheet);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var accCol = h.indexOf('accession code'), notesCol = h.indexOf('notes');

  var flagOut = [], detailOut = [], counts = {};

  for (var i = 1; i < data.length; i++) {
    var acc = String(data[i][accCol]).trim();
    var note = String(data[i][notesCol]).trim();
    var f = '', d = '';

    if (!acc && !note) { flagOut.push(['']); detailOut.push(['']); continue; }  // untouched row

    if (acc && acc.toUpperCase() !== 'N/A' && !/^accession_not_found$/i.test(acc)) {
      // ---- row HAS codes ----
      var codes = acc.split(';').map(function (s) { return s.trim(); }).filter(String);

      if (/=unclear \(mined/i.test(note)) {
        f = FLAGS.MINED_NO_CONTEXT;
        d = 'Code recovered by EBI text-mining from a paper with no downloadable full text. ' +
            'No surrounding sentence exists, so own-vs-reused cannot be determined from prose. ' +
            'Curator: check the repository record to see whether these authors deposited it.';
      } else if (/=unclear/i.test(note)) {
        f = FLAGS.PROV_UNCLEAR;
        d = 'Code found in full text, but the data-availability wording did not clearly ' +
            'indicate deposit vs reuse. Curator: read the DAS.';
      } else if (/paper-level\)/i.test(note)) {
        f = FLAGS.PROV_UNCLEAR;
        d = 'Provenance inferred from the paper as a whole, not from text near the accession. ' +
            'Lower confidence than a direct match.';
      } else {
        f = FLAGS.CLEAN;
        d = 'Code found in full text with clear deposit/reuse wording nearby.';
      }

      if (codes.length >= 4) {
        f = (f === FLAGS.CLEAN) ? FLAGS.MULTI_CODE : f + '+' + FLAGS.MULTI_CODE;
        d += ' ' + codes.length + ' codes on this row — curator may need to identify the primary.';
      }

    } else if (acc.toUpperCase() === 'N/A') {
      // ---- row is floor. Which kind? ----
      if (/publisher restricted/i.test(note)) {
        f = FLAGS.NO_FULLTEXT;
        d = 'PMC holds metadata only; publisher does not permit full-text download. ' +
            'Abstract and EBI-mined annotations both checked, no accession found. ' +
            'Not recoverable by this pipeline. Manual lookup of the published PDF would be needed.';
      } else if (/article\/supplement/i.test(note)) {
        f = FLAGS.SUPPLEMENT;
        d = 'Paper states the data is in supplementary files, which we never fetched. ' +
            'The accession may well be in a supplementary table. THIS IS RECOVERABLE — ' +
            'the single largest remaining opportunity in the corpus.';
      } else if (/on request/i.test(note)) {
        f = FLAGS.ON_REQUEST;
        d = 'Authors state data is available on request. True floor — no accession exists to find.';
      } else if (/no data generated/i.test(note)) {
        f = FLAGS.NO_DATA;
        d = 'Paper generated no new data. True floor.';
      } else if (/no accession in text/i.test(note)) {
        f = FLAGS.NO_ACCESSION;
        d = 'Full text read successfully, no accession code present in any recognised format. ' +
            'Either the paper genuinely has none, or it uses a repository outside our dictionary.';
      } else {
        f = FLAGS.NO_ACCESSION;
        d = 'N/A, reason not classified.';
      }

    } else {
      f = FLAGS.PENDING;
      d = 'Not yet resolved: ' + (note || 'no note');
    }

    flagOut.push([f]);
    detailOut.push([d]);
    counts[f] = (counts[f] || 0) + 1;
  }

  if (flagOut.length) {
    sheet.getRange(2, cols.flag + 1, flagOut.length, 1).setValues(flagOut);
    sheet.getRange(2, cols.detail + 1, detailOut.length, 1).setValues(detailOut);
  }

  var lines = [];
  for (var k in counts) lines.push('  ' + k + ': ' + counts[k]);
  lines.sort();
  var msg = 'STATIC FLAGGING DONE ✅\n\n' + lines.join('\n') +
            '\n\nNow run flagProvenanceConflicts() to add the second-opinion cross-check.';
  Logger.log(msg);
  SpreadsheetApp.getUi().alert(msg);
}

// ================= PASS 2: PROVENANCE CONFLICT. Network. Resumable. =================
// For every row where our regex confidently said own/reused, ask EPMC priority-by-date
// for a second opinion. Where they disagree AND the priority signal is trustworthy,
// flag it. This is the pass that surfaced rows 624/742 — EGA datasets our regex called
// "own" that 19-22 other papers had been citing for two years. Those are probably
// mislabelled reuse, and they are exactly what Sam wants a human to look at.
function flagProvenanceConflicts() {
  var t0 = Date.now();
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var cols = ensureFlagColumns_(sheet);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var pmidCol = h.indexOf('pubmed_id'), notesCol = h.indexOf('notes');

  var props = PropertiesService.getScriptProperties();
  var cursor = parseInt(props.getProperty('FLAGCONF') || '1', 10);

  var checked = 0, conflicts = 0, noSignal = 0, i = cursor;

  for (; i < data.length; i++) {
    if (Date.now() - t0 > CFG.TIME_BUDGET_MS) break;

    var note = String(data[i][notesCol]);
    if (note.indexOf('[auto]') !== 0) continue;

    var m = /([A-Z0-9._\/]+)=(own|reused)(?:\s*;|\s*$)/i.exec(note);   // confident labels only
    if (!m) continue;

    var code = m[1].toUpperCase(), regexProv = m[2].toLowerCase();
    var myPmid = String(data[i][pmidCol]).replace(/[^0-9]/g, '');
    if (!myPmid) continue;

    checked++;
    var v = priority_(code, myPmid);

    if (v.call === 'unknown') { noSignal++; }
    else if (v.call !== regexProv) {
      conflicts++;
      var existing = String(data[i][cols.flag] || '').trim();
      var f = existing && existing !== FLAGS.CLEAN
            ? existing + '+' + FLAGS.PROV_CONFLICT
            : FLAGS.PROV_CONFLICT;

      var d = 'CONFLICT. Text wording says "' + regexProv + '". But ' + v.n +
              ' papers mention ' + code + ' in Europe PMC, and the earliest (' + v.earliest +
              ') predates this paper (' + v.ourDate + '), which suggests "' + v.call + '". ' +
              'Curator: read the data-availability statement and decide.';

      sheet.getRange(i + 1, cols.flag + 1).setValue(f);
      sheet.getRange(i + 1, cols.detail + 1).setValue(d);
      Logger.log('CONFLICT row %s | %s | regex=%s priority=%s | %s mentioners, earliest %s vs ours %s',
                 i + 1, code, regexProv, v.call, v.n, v.earliest, v.ourDate);
    }

    if (checked % 50 === 0) {
      Logger.log('checked %s | conflicts %s | no signal %s | %ss',
                 checked, conflicts, noSignal, Math.round((Date.now() - t0) / 1000));
    }
    Utilities.sleep(250);
  }

  props.setProperty('FLAGCONF', String(i));
  var done = i >= data.length;
  if (done) props.deleteProperty('FLAGCONF');

  SpreadsheetApp.getUi().alert(
    (done ? 'CONFLICT PASS DONE ✅' : 'CHUNK done — RUN AGAIN') + '\n\n' +
    'rows checked: ' + checked + '\n' +
    'CONFLICTS flagged: ' + conflicts + '\n' +
    'no usable second opinion: ' + noSignal + '\n' +
    'next cursor row: ' + (i + 1)
  );
}

// Strict priority verdict. Returns 'unknown' unless the signal is genuinely trustworthy.
// Guards, learned the hard way from the probe:
//   - ZERO other mentioners = no signal. Do NOT call it 'own'. That default is what
//     wrongly flipped three regex-'reused' rows in the probe.
//   - pubYear-only dates get slammed to Jan 1 and fake a date gap. If either date is
//     approximate, demand a 2+ year gap before calling reuse.
function priority_(code, myPmid) {
  var hits = epmcHits_(code);
  if (hits === null || !hits.length) return { call: 'unknown', n: 0 };

  var ours = null, others = [];
  hits.forEach(function (x) {
    if (x.pmid && x.pmid === myPmid) ours = x;
    else others.push(x);
  });
  if (!others.length) return { call: 'unknown', n: hits.length };   // no comparison possible

  if (!ours) {
    var od = epmcDate_(myPmid);
    if (!od) return { call: 'unknown', n: hits.length };
    ours = od;
  }

  others.sort(function (a, b) { return a.date < b.date ? -1 : 1; });
  var first = others[0];

  var approx = ours.approx || first.approx;
  var ourY = parseInt(ours.date.slice(0, 4), 10);
  var firstY = parseInt(first.date.slice(0, 4), 10);

  if (approx) {
    if (ourY - firstY >= 2) return { call: 'reused', n: hits.length, ourDate: ours.date, earliest: first.date };
    return { call: 'unknown', n: hits.length };          // too noisy to judge
  }

  var call = (ours.date.slice(0, 7) <= first.date.slice(0, 7)) ? 'own' : 'reused';
  return { call: call, n: hits.length, ourDate: ours.date, earliest: first.date };
}

function epmcHits_(code) {
  var url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22' +
            encodeURIComponent(code) + '%22&format=json&pageSize=100&resultType=lite';
  for (var a = 0; a < 2; a++) {
    try {
      var r = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
      if (r.getResponseCode() !== 200) { Utilities.sleep(500); continue; }
      var res = (JSON.parse(r.getContentText()).resultList || {}).result || [];
      return res.map(function (x) {
        var full = x.firstPublicationDate || '';
        return {
          pmid: x.pmid ? String(x.pmid) : '',
          date: full || (x.pubYear ? x.pubYear + '-01-01' : ''),
          approx: !full && !!x.pubYear
        };
      }).filter(function (x) { return x.date; });
    } catch (e) { Utilities.sleep(400); }
  }
  return null;
}

function epmcDate_(pmid) {
  var url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:' + pmid +
            '%20AND%20SRC:MED&format=json&pageSize=1&resultType=lite';
  try {
    var r = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
    if (r.getResponseCode() !== 200) return null;
    var res = (JSON.parse(r.getContentText()).resultList || {}).result;
    if (!res || !res.length) return null;
    var full = res[0].firstPublicationDate || '';
    var d = full || (res[0].pubYear ? res[0].pubYear + '-01-01' : '');
    if (!d) return null;
    return { pmid: pmid, date: d, approx: !full };
  } catch (e) { return null; }
}

// ================= THE SCORECARD =================
function flagSummary() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var flagCol = h.indexOf('flag'), accCol = h.indexOf('accession code');
  if (flagCol < 0) { SpreadsheetApp.getUi().alert('No flag column. Run flagStatic() first.'); return; }

  var counts = {}, coded = 0, total = 0, codeCount = 0;
  for (var i = 1; i < data.length; i++) {
    var acc = String(data[i][accCol]).trim();
    if (!acc) continue;
    total++;
    if (acc.toUpperCase() !== 'N/A') {
      coded++;
      codeCount += acc.split(';').filter(function (s) { return s.trim(); }).length;
    }
    var f = String(data[i][flagCol]).trim() || 'UNFLAGGED';
    counts[f] = (counts[f] || 0) + 1;
  }

  var lines = [];
  for (var k in counts) lines.push(k + ': ' + counts[k]);
  lines.sort();

  var needsEyes = (counts[FLAGS.PROV_CONFLICT] || 0) +
                  (counts[FLAGS.PROV_UNCLEAR] || 0) +
                  (counts[FLAGS.MINED_NO_CONTEXT] || 0);

  var msg =
    'SCORECARD\n\n' +
    'rows answered: ' + total + '\n' +
    'rows with codes: ' + coded + ' (' + Math.round(100 * coded / total) + '%)\n' +
    'total codes: ' + codeCount + '\n\n' +
    lines.join('\n') + '\n\n' +
    'NEEDS CURATOR EYES: ' + needsEyes + '\n' +
    'RECOVERABLE LATER (supplement fetch): ' + (counts[FLAGS.SUPPLEMENT] || 0) + '\n' +
    'TRUE FLOOR: ' + ((counts[FLAGS.ON_REQUEST] || 0) + (counts[FLAGS.NO_DATA] || 0) +
                      (counts[FLAGS.NO_FULLTEXT] || 0));
  Logger.log(msg);
  SpreadsheetApp.getUi().alert(msg);
}

// ---------- resolve a row to an EPMC record ----------
// Returns {pmid, pmcid, title, abstract} or null. DOI first, then PMID.
function epmcRecord_(doi, pmid) {
  var queries = [];
  if (doi) {
    var d = doi.replace(/^https?:\/\/(dx\.)?doi\.org\//i, '').trim();
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
        if (!res || !res.length) break;
        var x = res[0];
        return {
          pmid:     x.pmid ? String(x.pmid) : '',
          pmcid:    x.pmcid ? String(x.pmcid) : '',
          title:    x.title || '',
          abstract: x.abstractText || ''
        };
      } catch (e) { Utilities.sleep(400); }
    }
  }
  return null;
}

// ---------- EBI mined annotations, keyed by PMID (MED) not PMCID ----------
function fetchAnnotationsMed_(pmids) {
  var out = {};
  if (!pmids || !pmids.length) return out;

  var ids = pmids.map(function (p) { return 'MED%3A' + String(p).replace(/[^0-9]/g, ''); }).join(',');
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
      DICT.forEach(function (entry) {
        var repo = entry[0], re = entry[1];
        re.lastIndex = 0;
        var m = re.exec(exact);
        if (!m || m[0].length !== exact.length) return;
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
