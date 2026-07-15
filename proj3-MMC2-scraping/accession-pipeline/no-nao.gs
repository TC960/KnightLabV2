/*  NON-OA HANDLING — v2, with EPMC annotation mining
 *
 *  CONFIRMED BY PROBE:
 *    EBI text-mines articles it cannot redistribute and exposes extracted accession
 *    numbers via the annotations API. PMC9757091 (no downloadable full text) returned
 *    PRJNA788785 / subType BioProject. A real code from a paper we'd written off.
 *
 *    But: 3 of 5 probed papers returned an empty annotation array, and PMC9986680
 *    returned only RefSNP variant IDs (rs6667202) which are NOT data deposits.
 *    So we do NOT trust subType blindly — every mined string is run through the same
 *    DICT regexes as everything else. rs-numbers match nothing and drop out for free.
 *
 *  TWO CHANNELS, UNIONED:
 *    (a) EPMC annotations API — codes EBI mined, no full text needed. Batched 8/req.
 *    (b) NCBI front matter    — the ~12KB of abstract+metadata we already download
 *                               and currently throw away.
 *    A code found by either channel counts. Deduped across both.
 *
 *  Depends on runExtraction.gs for: CFG, RETRY_MARK, REVIEW_MARK, DICT, OWN, REUSE,
 *  extractCodes_, writeResult_.
 */

var NONOA_MARK = '[auto] N/A - no full text (publisher restricted)';
var NONOA_RE   = /does not allow downloading of the full text/i;
var ANN_BATCH  = 8;   // annotationsByArticleIds takes several IDs per request

// ---------- fetch: non-OA is an outcome, not a failure ----------
function fetchArticle(pmcRaw) {
  var digits = String(pmcRaw).replace(/[^0-9]/g, '');
  if (!digits) return { kind: 'FAIL', xml: '' };

  var epmc = 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC' + digits + '/fullTextXML';
  var ncbi = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=' +
             digits + '&rettype=full&retmode=xml' + (CFG.NCBI_KEY ? '&api_key=' + CFG.NCBI_KEY : '');

  for (var a = 0; a < 2; a++) {
    try {
      var r = UrlFetchApp.fetch(epmc, { muteHttpExceptions: true });
      var c = r.getResponseCode();
      if (c === 200) {
        var t = r.getContentText();
        if (/<body[\s>]/i.test(t)) return { kind: 'FULL', xml: t };
        break;
      }
      if (c === 404) break;   // not in EPMC's OA set — expected for non-OA
    } catch (e) { Utilities.sleep(400); }
  }

  for (var b = 0; b < 2; b++) {
    try {
      var r2 = UrlFetchApp.fetch(ncbi, { muteHttpExceptions: true });
      if (r2.getResponseCode() !== 200) { Utilities.sleep(600); continue; }
      var t2 = r2.getContentText();
      if (/<body[\s>]/i.test(t2)) return { kind: 'FULL', xml: t2 };
      if (NONOA_RE.test(t2) || /<article[\s>]/i.test(t2)) return { kind: 'FRONT', xml: t2 };
    } catch (e2) { Utilities.sleep(400); }
  }
  return { kind: 'FAIL', xml: '' };
}

// ---------- channel (a): EPMC mined annotations, batched ----------
// Input: array of PMCID strings. Output: { "PMC9757091": [{code,repo,prov}, ...], ... }
function fetchAnnotations(pmcids) {
  var out = {};
  if (!pmcids || !pmcids.length) return out;

  var ids = pmcids.map(function (p) {
    return 'PMC%3A' + String(p).replace(/[^0-9]/g, '');
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
    var pmcid = String(rec.pmcid || '').toUpperCase();
    if (!pmcid) return;
    var codes = [], seen = {};

    (rec.annotations || []).forEach(function (ann) {
      var exact = String(ann.exact || '').trim();
      if (!exact) return;

      // Run the mined string through OUR dictionary. This is the whole safety net:
      // rs6667202 (RefSNP) matches no DICT entry and is silently dropped. No subType
      // whitelist to maintain, and the dictionary stays the single source of truth.
      DICT.forEach(function (entry) {
        var repo = entry[0], re = entry[1];
        re.lastIndex = 0;
        var m = re.exec(exact);
        if (!m) return;
        if (m[0].length !== exact.length) return;   // must match the WHOLE mined token

        var code = m[0].toUpperCase();
        if (seen[code]) return;
        seen[code] = true;

        // provenance from mined context when EBI supplies any — often it supplies none
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

    if (codes.length) out[pmcid] = codes;
  });

  return out;
}

// ---------- the sweep ----------
function sweepNonOA() {
  var t0 = Date.now();
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var pmcCol = h.indexOf('pmc_id'), accCol = h.indexOf('accession code'), notesCol = h.indexOf('notes');

  var props = PropertiesService.getScriptProperties();
  var cursor = parseInt(props.getProperty('NONOA') || '1', 10);

  // gather this chunk's rows first so the annotation calls can be batched
  var targets = [], i = cursor;
  for (; i < data.length && targets.length < 200; i++) {
    var note = String(data[i][notesCol]).trim();
    if (note !== RETRY_MARK && note !== REVIEW_MARK) continue;
    var pmc = String(data[i][pmcCol]).trim();
    if (!pmc) continue;
    targets.push({ row: i + 1, pmc: pmc });
  }
  var nextCursor = i;

  Logger.log('=== sweepNonOA START — cursor %s, %s targets this chunk ===', cursor, targets.length);

  // --- batched annotation lookups: 691 rows becomes ~87 requests, not 691 ---
  var annMap = {};
  for (var b = 0; b < targets.length; b += ANN_BATCH) {
    if (Date.now() - t0 > CFG.TIME_BUDGET_MS) break;
    var slice = targets.slice(b, b + ANN_BATCH).map(function (t) { return t.pmc; });
    var got = fetchAnnotations(slice);
    for (var k in got) annMap[k] = got[k];
    Utilities.sleep(CFG.PACE_MS);
  }
  Logger.log('annotations: %s of %s papers came back with usable codes',
             Object.keys(annMap).length, targets.length);

  // --- walk the rows, union annotation codes with front-matter codes ---
  var fromAnn = 0, fromFront = 0, both = 0, nonOA = 0, stillFail = 0, done = 0;

  for (var j = 0; j < targets.length; j++) {
    if (Date.now() - t0 > CFG.TIME_BUDGET_MS) { nextCursor = targets[j].row - 1; break; }
    var tgt = targets[j];
    done++;

    var key = 'PMC' + String(tgt.pmc).replace(/[^0-9]/g, '');
    var annCodes = annMap[key] || [];

    var art = fetchArticle(tgt.pmc);
    var frontCodes = [], reason = 'no accession in text';
    if (art.kind !== 'FAIL') {
      var plain = art.xml.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
      var res = extractCodes_(plain);
      frontCodes = res.codes;
      reason = res.reason;
    }

    // union — annotation codes first, dedup on the code string
    var merged = [], seen = {};
    annCodes.concat(frontCodes).forEach(function (c) {
      if (seen[c.code]) return;
      seen[c.code] = true;
      merged.push(c);
    });

    if (merged.length) {
      writeResult_(sheet, tgt.row, accCol, notesCol, { codes: merged, reason: '' });
      if (annCodes.length && frontCodes.length) both++;
      else if (annCodes.length) fromAnn++;
      else fromFront++;
      Logger.log('  row %s RECOVERED [%s]: %s', tgt.row,
                 (annCodes.length ? (frontCodes.length ? 'both' : 'annotations') : 'front matter'),
                 merged.map(function (c) { return c.code + '=' + c.prov; }).join('; '));
    } else if (art.kind === 'FRONT') {
      sheet.getRange(tgt.row, accCol + 1).setValue('N/A');
      sheet.getRange(tgt.row, notesCol + 1).setValue(NONOA_MARK);
      nonOA++;
    } else if (art.kind === 'FULL') {
      sheet.getRange(tgt.row, accCol + 1).setValue('N/A');
      sheet.getRange(tgt.row, notesCol + 1).setValue('[auto] ' + reason);
      nonOA++;
    } else {
      stillFail++;   // genuinely unreachable — leave the mark. THIS is the real retry pool.
    }

    if (done % 25 === 0) {
      Logger.log('swept %s | ann %s · front %s · both %s · N/A %s · unreachable %s | %ss',
                 done, fromAnn, fromFront, both, nonOA, stillFail,
                 Math.round((Date.now() - t0) / 1000));
    }
    Utilities.sleep(CFG.PACE_MS);
  }

  props.setProperty('NONOA', String(nextCursor));
  var allDone = nextCursor >= data.length;
  if (allDone) props.deleteProperty('NONOA');

  var recovered = fromAnn + fromFront + both;
  SpreadsheetApp.getUi().alert(
    (allDone ? 'NON-OA SWEEP DONE ✅' : 'CHUNK done — RUN AGAIN to continue') + '\n\n' +
    'swept: ' + done + '\n' +
    '  RECOVERED: ' + recovered + '\n' +
    '    EPMC annotations only: ' + fromAnn + '\n' +
    '    front matter only:     ' + fromFront + '\n' +
    '    found by both:         ' + both + '\n' +
    '  N/A (publisher restricted): ' + nonOA + '\n' +
    '  still unreachable:          ' + stillFail + '\n' +
    'next cursor row: ' + (nextCursor + 1)
  );
}

// ---------- optional, read-only: does EBI find codes our regexes missed? ----------
// EBI mines OA papers too. This checks the rows we already marked N/A and logs what
// the annotations API would have added. Writes NOTHING. Run it, read the log, decide.
// If it lights up, our dictionary or our full-text extraction has a gap worth finding.
function auditAnnotationsVsOurs() {
  var t0 = Date.now();
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var pmcCol = h.indexOf('pmc_id'), accCol = h.indexOf('accession code');

  var batch = [], rowsFor = {}, wouldAdd = 0, checked = 0;

  for (var i = 1; i < data.length; i++) {
    if (Date.now() - t0 > CFG.TIME_BUDGET_MS) break;
    if (String(data[i][accCol]).trim().toUpperCase() !== 'N/A') continue;   // the N/A floor only
    var pmc = String(data[i][pmcCol]).trim();
    if (!pmc) continue;

    batch.push(pmc);
    rowsFor['PMC' + pmc.replace(/[^0-9]/g, '')] = i + 1;

    if (batch.length === ANN_BATCH) {
      var got = fetchAnnotations(batch);
      for (var k in got) {
        wouldAdd++;
        Logger.log('row %s (%s) is N/A — annotations say: %s',
                   rowsFor[k], k, got[k].map(function (c) { return c.code; }).join('; '));
      }
      checked += batch.length;
      batch = []; rowsFor = {};
      Utilities.sleep(CFG.PACE_MS);
    }
  }

  Logger.log('=== audit: checked %s N/A rows, annotations would add codes to %s ===', checked, wouldAdd);
  SpreadsheetApp.getUi().alert('Audit done.\nChecked ' + checked + ' N/A rows.\n' +
    'Annotations would add codes to ' + wouldAdd + ' of them.\nSee the log.');
}