/**
 * DISTRIBUTION — where does every row stand, and what's inside each bucket.
 * Level 1: top-level buckets (has code / N/A / fetch-failed / review / untouched).
 * Level 2: drill-down of the two rich buckets:
 *     - HAS CODE  -> by source, by provenance, by repository (per-code)
 *     - N/A       -> by reason
 * Read-only. Writes "_distribution" + a summary alert.
 */
function distribution() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName('articles.csv');
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var pmcCol = h.indexOf('pmc_id'), accCol = h.indexOf('accession code'), notesCol = h.indexOf('notes');

  // ---- repository classifier for a single code token ----
  var repoOf = function (t) {
    t = t.toUpperCase();
    if (/^PRJ/.test(t)) return 'BioProject';
    if (/^SAM/.test(t)) return 'BioSample';
    if (/^[SED]R[APRSX]/.test(t)) return 'SRA/ENA/DDBJ';
    if (/^G(SE|SM|PL|DS)/.test(t)) return 'GEO';
    if (/^PHS/.test(t)) return 'dbGaP';
    if (/^HRA/.test(t)) return 'GSA-Human';
    if (/^KAP/.test(t)) return 'GSA-KAP';
    if (/^CR[ARX]/.test(t)) return 'GSA';
    if (/^OE/.test(t)) return 'NODE';
    if (/^CN[PSXR]/.test(t)) return 'CNGB';
    if (/^MG[MP]/.test(t)) return 'MG-RAST';
    if (/^MTBLS/.test(t)) return 'MetaboLights';
    if (/^PXD/.test(t)) return 'PRIDE';
    if (/^EGA/.test(t)) return 'EGA';
    if (/^E-/.test(t)) return 'ArrayExpress';
    if (/^S-/.test(t)) return 'BioStudies';
    if (/FIGSHARE/.test(t)) return 'figshare';
    if (/ZENODO/.test(t)) return 'Zenodo';
    if (/DRYAD/.test(t)) return 'Dryad';
    return 'Other/DOI';
  };
  var inc = function (o, k) { o[k] = (o[k] || 0) + 1; };

  var buckets = { hasCode: 0, na: 0, fetchFail: 0, review: 0, untouched: 0 };
  var source = {}, provenance = {}, repo = {}, naReason = {}, untouchedSplit = { withPmc: 0, noPmc: 0 };
  var total = 0;

  for (var i = 1; i < data.length; i++) {
    var row = data[i];
    if (row.every(function (c) { return String(c).trim() === ''; })) continue;
    total++;
    var acc = String(row[accCol]).trim();
    var note = String(row[notesCol]).trim();
    var pmc = String(row[pmcCol]).trim();
    var accU = acc.toUpperCase();

    // priority-ordered classification
    if (note.indexOf('[review]') === 0) { buckets.review++; continue; }
    if (note === '[auto] fetch failed - retry') { buckets.fetchFail++; continue; }

    var hasRealCode = acc !== '' && accU !== 'N/A' && !/^ACCESSION_NOT_FOUND$/i.test(acc);

    if (hasRealCode) {
      buckets.hasCode++;
      // source
      if (note.indexOf('[auto-oa]') === 0) inc(source, 'auto (open-access path)');
      else if (note.indexOf('[auto]') === 0) inc(source, 'auto (PMC path)');
      else inc(source, 'human-entered');
      // provenance per code (from note tags), else untagged
      var provTags = note.match(/=(own|reused|unclear)/gi);
      if (provTags) provTags.forEach(function (p) { inc(provenance, p.slice(1).toLowerCase()); });
      else inc(provenance, 'human (untagged)');
      // repository per code token
      acc.split(/[;,\s]+/).filter(Boolean).forEach(function (tok) { inc(repo, repoOf(tok)); });
      continue;
    }

    if (accU === 'N/A' || note.indexOf('[auto]') === 0 || note.indexOf('[auto-oa]') === 0) {
      buckets.na++;
      var reason = note.replace(/^\[auto(-oa)?\]\s*/i, '').trim() || 'unspecified';
      inc(naReason, reason);
      continue;
    }

    // nothing written yet
    buckets.untouched++;
    if (pmc) untouchedSplit.withPmc++; else untouchedSplit.noPmc++;
  }

  // ---- write _distribution tab ----
  var out = [];
  var pct = function (n) { return total ? (100 * n / total).toFixed(1) + '%' : '0%'; };
  out.push(['LEVEL 1 — TOP BUCKETS', 'count', 'share']);
  out.push(['Has accession code', buckets.hasCode, pct(buckets.hasCode)]);
  out.push(['N/A (no code, with reason)', buckets.na, pct(buckets.na)]);
  out.push(['Fetch failed (pending retry)', buckets.fetchFail, pct(buckets.fetchFail)]);
  out.push(['Review (manual check)', buckets.review, pct(buckets.review)]);
  out.push(['Untouched (blank)', buckets.untouched, pct(buckets.untouched)]);
  out.push(['TOTAL', total, '100%']);
  out.push(['', '', '']);

  var dump = function (title, obj) {
    out.push([title, '', '']);
    Object.keys(obj).sort(function (a, b) { return obj[b] - obj[a]; })
      .forEach(function (k) { out.push(['   ' + k, obj[k], '']); });
    out.push(['', '', '']);
  };
  dump('LEVEL 2 — HAS CODE · by source', source);
  dump('LEVEL 2 — HAS CODE · by provenance (per code)', provenance);
  dump('LEVEL 2 — HAS CODE · by repository (per code)', repo);
  dump('LEVEL 2 — N/A · by reason', naReason);
  out.push(['LEVEL 2 — UNTOUCHED · split', '', '']);
  out.push(['   has pmc_id (should be ~0 after pass 1)', untouchedSplit.withPmc, '']);
  out.push(['   no pmc_id (phase-2 territory)', untouchedSplit.noPmc, '']);

  var d = ss.getSheetByName('_distribution');
  if (d) d.clear(); else d = ss.insertSheet('_distribution');
  d.getRange(1, 1, out.length, 3).setValues(out);
  d.setColumnWidth(1, 340); d.setColumnWidth(2, 90); d.setColumnWidth(3, 80);

  SpreadsheetApp.getUi().alert(
    'DISTRIBUTION (' + total + ' rows) -> tab "_distribution"\n\n' +
    'Has code:        ' + buckets.hasCode + '  (' + pct(buckets.hasCode) + ')\n' +
    'N/A + reason:    ' + buckets.na + '  (' + pct(buckets.na) + ')\n' +
    'Fetch failed:    ' + buckets.fetchFail + '\n' +
    'Review:          ' + buckets.review + '\n' +
    'Untouched:       ' + buckets.untouched +
      '  (pmc ' + untouchedSplit.withPmc + ' / no-pmc ' + untouchedSplit.noPmc + ')\n\n' +
    'Full drill-down (source / provenance / repository / N/A reason) is in the tab.'
  );
}