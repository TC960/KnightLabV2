/**
 * Refines the audit: separates the previous pipeline's ACCESSION_NOT_FOUND
 * sentinel from cells that hold a real-but-unrecognized format.
 * Dumps ONLY the genuinely-unknown cells (not the sentinel) so you can
 * see what formats the dictionary is still missing.
 * Read-only except for the scratch dump tab.
 */

function auditUnknownFormats() {
  const SHEET_NAME = '';
  const ACC_HEADER = 'accession code';
  const USE_NTH_ACC = 1;

  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = SHEET_NAME ? ss.getSheetByName(SHEET_NAME) : ss.getActiveSheet();
  const data = sheet.getDataRange().getValues();
  const headers = data[0].map(h => String(h).trim().toLowerCase());

  let accCol = -1, seen = 0;
  for (let c = 0; c < headers.length; c++) {
    if (headers[c] === ACC_HEADER) { seen++; if (seen === USE_NTH_ACC) { accCol = c; break; } }
  }
  if (accCol < 0) { SpreadsheetApp.getUi().alert('acc col not found'); return; }

  const PATTERNS = [
    ['BioProject',    /\bPRJ(?:EB|NA|DB|CA)\d+\b/i],
    ['BioSample',     /\bSAM(?:EA|N|D)\d+\b/i],
    ['SRA/ENA/DDBJ',  /\b[SED]R[APRSX]\d+\b/i],
    ['GEO',           /\bG(?:SE|SM|PL|DS)\d+\b/i],
    ['dbGaP',         /\bphs\d+(?:\.v\d+\.p\d+)?\b/i],
    ['GSA (CRA)',     /\bCR[ARX]\d+\b/i],
    ['NODE (OEP)',    /\bOEP\d+\b/i],
    ['CNGB (CNP)',    /\bCN[PSXR]\d+\b/i],
    ['MG-RAST',       /\bmg[mp]\d+(?:\.\d+)?\b/i],
    ['MetaboLights',  /\bMTBLS\d+\b/i],
    ['PRIDE',         /\bPXD\d+\b/i],
    ['EGA',           /\bEGA[SD]\d+\b/i],
    ['ERP/ERS/ERX',   /\bER[PSX]\d+\b/i],
    ['ArrayExpress',  /\bE-[A-Z]{4}-\d+\b/i],       // E-MTAB-11957, E-GEOD-..., etc.
    ['Zenodo',        /10\.5281\/zenodo\.\d+/i],
    ['Figshare',      /10\.6084\/m9\.figshare\.\d+/i],
    ['Dryad',         /10\.506\d\/dryad\.\w+/i],
    ['Qiita (URL)',   /qiita\.ucsd\.edu\/study\/\S*\d+/i],
  ];

  // everything a human OR the old pipeline used to mean "nothing here"
  const NOT_FOUND = /^(n\/?a|none|no accession(?: code)?(?: found)?|not found|not available|n\.a\.?|-{1,}|null|accession_not_found)$/i;

  let empty = 0, sentinel = 0, otherNotFound = 0, matches = 0, unknown = 0, total = 0;
  const unknownSamples = [];
  const unknownFreq = {};   // count distinct unknown values

  for (let i = 1; i < data.length; i++) {
    const row = data[i];
    if (row.every(c => String(c).trim() === '')) continue;
    total++;
    const raw = String(row[accCol]).trim();

    if (raw === '') { empty++; continue; }
    if (/^accession_not_found$/i.test(raw)) { sentinel++; continue; }
    if (NOT_FOUND.test(raw)) { otherNotFound++; continue; }

    const tokens = raw.split(/[\s,;\/|]+/).filter(Boolean);
    let anyMatch = false;
    for (const tok of tokens) {
      for (const [, re] of PATTERNS) { if (re.test(tok) || re.test(raw)) { anyMatch = true; break; } }
      if (anyMatch) break;
    }

    if (anyMatch) { matches++; }
    else {
      unknown++;
      const key = raw.slice(0, 60);
      unknownFreq[key] = (unknownFreq[key] || 0) + 1;
      if (unknownSamples.length < 100) unknownSamples.push('[row ' + (i+1) + '] ' + raw.slice(0, 100));
    }
  }

  const pct = n => total ? (100 * n / total).toFixed(1) + '%' : '0%';
  const report =
    'REFINED AUDIT (' + total + ' rows)\n\n' +
    'Empty:                     ' + empty + '  (' + pct(empty) + ')\n' +
    'ACCESSION_NOT_FOUND:       ' + sentinel + '  (' + pct(sentinel) + ')  <- old pipeline misses\n' +
    'Other "not found":         ' + otherNotFound + '  (' + pct(otherNotFound) + ')\n' +
    'Matches known format:      ' + matches + '  (' + pct(matches) + ')\n' +
    'Genuinely UNKNOWN format:  ' + unknown + '  (' + pct(unknown) + ')  <- new formats to add\n\n' +
    'TRUE need-to-fetch = empty + sentinel + other = ' +
      (empty + sentinel + otherNotFound) + '  (' + pct(empty + sentinel + otherNotFound) + ')';

  Logger.log(report);

  const dumpName = '_acc_unknown_formats';
  let dump = ss.getSheetByName(dumpName);
  if (dump) dump.clear(); else dump = ss.insertSheet(dumpName);
  dump.getRange(1, 1, 1, 2).setValues([['unknown cell (sample)', 'count of this exact value']]);
  const freqRows = Object.entries(unknownFreq).sort((a,b)=>b[1]-a[1]);
  if (freqRows.length) dump.getRange(2, 1, freqRows.length, 2).setValues(freqRows);

  SpreadsheetApp.getUi().alert(report + '\n\nDistinct unknown values written to "' + dumpName + '".');
}