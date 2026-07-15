/*  PROVENANCE VIA THE REGISTRY  +  dbGaP DEDUP FIX
 *
 *  PROBLEM 1 — duplicate dbGaP codes.
 *    sweepNonOA is writing both PHS001768 (mined, bare) and PHS001768.V1.P1
 *    (front matter, versioned). Same study, two rows in the cell, contradictory
 *    provenance. dedupCodes_() collapses them on the phs number, keeping the
 *    versioned form (more specific) and the strongest provenance available.
 *
 *  PROBLEM 2 — 'unclear (mined, no context)' on ~75 recovered codes.
 *    EBI hands back a bare accession with no surrounding sentence, so there is
 *    zero linguistic signal to classify own-vs-reused. Guessing would be worse
 *    than useless in a dataset Sam is going to trust.
 *
 *    So: ask the REGISTRY instead of the prose. Every BioProject and GEO series
 *    records which publications it belongs to. NCBI elink exposes that mapping.
 *      - project links to THIS paper's PMID     -> the authors deposited it   -> own
 *      - project links only to OTHER PMIDs      -> they cited someone's data  -> reused
 *      - project links to nothing               -> stay honest, leave unclear
 *
 *    This is ground truth from the deposit record, not inference from wording.
 *    It's a STRONGER signal than the OWN/REUSE regexes, and worth considering as
 *    a cross-check on the regex-derived provenance later.
 *
 *  HONEST CAVEATS, put these in the methods note:
 *    - A project with no linked publication stays 'unclear'. Absence of a link is
 *      not evidence of reuse; the registry lags, and links are often never added.
 *    - 'reused (registry)' means the project links to a different paper. Usually
 *      that means reuse. It can also mean the authors deposited under a project
 *      registered to an earlier companion paper of their own. Treat as high-
 *      confidence but not infallible.
 *    - Only PRJ* and GSE* are resolvable this way. SRP/ERP/SRR/SAMN/PHS stay as-is.
 *
 *  Costs 2 NCBI calls per code. ~75 codes = ~150 calls. Cheap.
 *  Depends on runExtraction.gs for CFG.
 */

// ---------- dedup: collapse dbGaP versions, keep the best provenance ----------
var PROV_RANK = { 'own': 3, 'reused': 3, 'own (paper-level)': 2, 'reused (paper-level)': 2,
                  'unclear (mixed)': 1, 'unclear': 0, 'unclear (mined, no context)': 0 };

function dedupCodes_(codes) {
  var byKey = {};

  codes.forEach(function (c) {
    // dbGaP: PHS000265 and PHS000265.V3.P1 are the same study. Collapse on the phs number.
    var m = /^(PHS\d{6})/i.exec(c.code);
    var key = m ? m[1].toUpperCase() : c.code.toUpperCase();

    var prev = byKey[key];
    if (!prev) { byKey[key] = c; return; }

    // keep the more specific code string (versioned beats bare)
    var better = (c.code.length > prev.code.length) ? c : prev;
    // keep the strongest provenance from either
    var pc = PROV_RANK[c.prov] === undefined ? 0 : PROV_RANK[c.prov];
    var pp = PROV_RANK[prev.prov] === undefined ? 0 : PROV_RANK[prev.prov];
    var prov = (pc > pp) ? c.prov : prev.prov;

    byKey[key] = { code: better.code, repo: better.repo, prov: prov };
  });

  var out = [];
  for (var k in byKey) out.push(byKey[k]);
  return out;
}

// ---------- registry lookups ----------
// BioProject accession -> the PMIDs that project is linked to.
function bioprojectPmids(acc) {
  var uid = esearchUid_('bioproject', acc + '[Project Accession]');
  if (!uid) return [];
  return elinkPmids_('bioproject', uid);
}

// GEO series (GSE...) -> linked PMIDs, via the gds database.
function geoPmids(acc) {
  var uid = esearchUid_('gds', acc + '[Accession]');
  if (!uid) return [];
  return elinkPmids_('gds', uid);
}

function esearchUid_(db, term) {
  var url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=' + db +
            '&term=' + encodeURIComponent(term) + '&retmode=json' +
            (CFG.NCBI_KEY ? '&api_key=' + CFG.NCBI_KEY : '');
  try {
    var r = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
    if (r.getResponseCode() !== 200) return '';
    var j = JSON.parse(r.getContentText());
    var ids = j && j.esearchresult && j.esearchresult.idlist;
    return (ids && ids.length) ? String(ids[0]) : '';
  } catch (e) { return ''; }
}

function elinkPmids_(dbfrom, uid) {
  var url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=' + dbfrom +
            '&db=pubmed&id=' + uid + '&retmode=json' +
            (CFG.NCBI_KEY ? '&api_key=' + CFG.NCBI_KEY : '');
  try {
    var r = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
    if (r.getResponseCode() !== 200) return [];
    var j = JSON.parse(r.getContentText());
    var out = [];
    (j.linksets || []).forEach(function (ls) {
      (ls.linksetdbs || []).forEach(function (db) {
        if (db.dbto !== 'pubmed') return;
        (db.links || []).forEach(function (id) { out.push(String(id)); });
      });
    });
    return out;
  } catch (e) { return []; }
}

// ---------- the resolver ----------
// Walks rows whose notes contain an 'unclear' provenance, resolves what it can
// against the registry, rewrites the notes. Resumable.
function resolveProvenanceViaRegistry() {
  var t0 = Date.now();
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var pmidCol = h.indexOf('pubmed_id'), notesCol = h.indexOf('notes');

  var props = PropertiesService.getScriptProperties();
  var cursor = parseInt(props.getProperty('PROVREG') || '1', 10);

  var toOwn = 0, toReused = 0, stayed = 0, skipped = 0, i = cursor;
  Logger.log('=== resolveProvenanceViaRegistry START — cursor %s ===', cursor);

  for (; i < data.length; i++) {
    if (Date.now() - t0 > CFG.TIME_BUDGET_MS) break;

    var note = String(data[i][notesCol]);
    if (note.indexOf('[auto]') !== 0) continue;
    if (note.indexOf('unclear') < 0) continue;          // only the unresolved ones

    var myPmid = String(data[i][pmidCol]).replace(/[^0-9]/g, '');
    var sheetRow = i + 1;

    // notes look like: [auto] PRJNA656590=unclear (mined, no context); SRP1234=own
    var parts = note.replace(/^\[auto\]\s*/, '').split(';');
    var changed = false;

    var rebuilt = parts.map(function (p) {
      var seg = p.trim();
      if (!seg) return '';
      var eq = seg.lastIndexOf('=');
      if (eq < 0) return seg;
      var code = seg.slice(0, eq).trim().toUpperCase();
      var prov = seg.slice(eq + 1).trim();
      if (prov.indexOf('unclear') !== 0) return seg;    // already resolved, leave it

      var pmids = [];
      if (/^PRJ(EB|NA|DB|CA)\d+$/i.test(code))      pmids = bioprojectPmids(code);
      else if (/^GSE\d+$/i.test(code))              pmids = geoPmids(code);
      else { skipped++; return seg; }                   // not resolvable this way

      Utilities.sleep(350);   // 2 calls already made above; stay under 3/s

      if (!pmids.length) { stayed++; return code + '=' + prov; }

      if (myPmid && pmids.indexOf(myPmid) >= 0) {
        changed = true; toOwn++;
        return code + '=own (registry)';
      }
      changed = true; toReused++;
      return code + '=reused (registry)';
    }).filter(function (s) { return s; });

    if (changed) {
      var newNote = '[auto] ' + rebuilt.join('; ');
      sheet.getRange(sheetRow, notesCol + 1).setValue(newNote);
      Logger.log('  row %s -> %s', sheetRow, newNote);
    }
  }

  props.setProperty('PROVREG', String(i));
  var done = i >= data.length;
  if (done) props.deleteProperty('PROVREG');

  SpreadsheetApp.getUi().alert(
    (done ? 'PROVENANCE RESOLVE DONE ✅' : 'CHUNK done — RUN AGAIN to continue') + '\n\n' +
    'resolved to own:    ' + toOwn + '\n' +
    'resolved to reused: ' + toReused + '\n' +
    'no registry link (left unclear): ' + stayed + '\n' +
    'not resolvable (SRP/SAMN/PHS/etc): ' + skipped + '\n' +
    'next cursor row: ' + (i + 1)
  );
}

// ---------- one-off: clean up the dbGaP duplicates already written ----------
// Rewrites existing notes + accession cells through dedupCodes_. No fetching.
function fixDuplicateCodes() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(CFG.SHEET);
  var data = sheet.getDataRange().getValues();
  var h = data[0].map(function (x) { return String(x).trim().toLowerCase(); });
  var accCol = h.indexOf('accession code'), notesCol = h.indexOf('notes');
  var n = 0;

  for (var i = 1; i < data.length; i++) {
    var note = String(data[i][notesCol]);
    if (note.indexOf('[auto]') !== 0 || note.indexOf('=') < 0) continue;

    var codes = note.replace(/^\[auto\]\s*/, '').split(';').map(function (p) {
      var seg = p.trim();
      var eq = seg.lastIndexOf('=');
      if (eq < 0) return null;
      return { code: seg.slice(0, eq).trim().toUpperCase(), repo: '', prov: seg.slice(eq + 1).trim() };
    }).filter(function (c) { return c && c.code; });

    if (codes.length < 2) continue;
    var deduped = dedupCodes_(codes);
    if (deduped.length === codes.length) continue;      // nothing collapsed

    sheet.getRange(i + 1, accCol + 1).setValue(deduped.map(function (c) { return c.code; }).join('; '));
    sheet.getRange(i + 1, notesCol + 1).setValue('[auto] ' +
      deduped.map(function (c) { return c.code + '=' + c.prov; }).join('; '));
    n++;
    Logger.log('row %s: %s -> %s', i + 1,
               codes.map(function (c) { return c.code; }).join('; '),
               deduped.map(function (c) { return c.code; }).join('; '));
  }
  SpreadsheetApp.getUi().alert('Collapsed duplicate codes on ' + n + ' rows.');
}
function debugRegistry() {
  var tests = ['PRJNA656590', 'PRJNA788785', 'PRJNA646610', 'GSE147600'];

  tests.forEach(function (acc) {
    Logger.log('========== %s ==========', acc);
    var db = /^GSE/i.test(acc) ? 'gds' : 'bioproject';

    // three candidate query forms — find out which one actually resolves a UID
    var terms = /^GSE/i.test(acc)
      ? [acc, acc + '[Accession]', acc + '[GEO Accession]']
      : [acc, acc + '[Project Accession]', acc + '[BioProject]', acc + '[All Fields]'];

    terms.forEach(function (term) {
      var url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=' + db +
                '&term=' + encodeURIComponent(term) + '&retmode=json';
      try {
        var r = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
        var j = JSON.parse(r.getContentText());
        var ids = (j.esearchresult && j.esearchresult.idlist) || [];
        Logger.log('  esearch "%s" -> HTTP %s | count %s | ids: %s',
                   term, r.getResponseCode(),
                   (j.esearchresult && j.esearchresult.count), ids.join(','));

        if (ids.length) {
          var el = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=' + db +
                   '&db=pubmed&id=' + ids[0] + '&retmode=json';
          var r2 = UrlFetchApp.fetch(el, { muteHttpExceptions: true });
          Logger.log('    elink -> HTTP %s | %s', r2.getResponseCode(),
                     r2.getContentText().slice(0, 400));
        }
      } catch (e) { Logger.log('  "%s" THREW %s', term, e); }
      Utilities.sleep(400);
    });
  });
}