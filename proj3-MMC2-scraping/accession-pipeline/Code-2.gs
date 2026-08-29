/**
 * For rows that NEED an accession, how reachable are they?
 * Splits into three groups:
 *   - sentinel  = old pipeline wrote ACCESSION_NOT_FOUND
 *   - blank     = never attempted
 *   - filled    = already has a valid-looking accession (context only)
 * and for each group reports how many have pmc_id / pubmed_id / doi.
 *
 * This tells us which fetch path the prototype must support.
 * Read-only.
 */

function auditIdentifiers() {
  const SHEET_NAME = '';
  const USE_NTH_ACC = 1;

  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = SHEET_NAME ? ss.getSheetByName(SHEET_NAME) : ss.getActiveSheet();
  const data = sheet.getDataRange().getValues();
  const headers = data[0].map(h => String(h).trim().toLowerCase());

  const nthCol = (name, n) => {
    let seen = 0;
    for (let c = 0; c < headers.length; c++) {
      if (headers[c] === name) { seen++; if (seen === n) return c; }
    }
    return -1;
  };
  const col = name => headers.indexOf(name);

  const accCol  = nthCol('accession code', USE_NTH_ACC);
  const pmcCol  = col('pmc_id');
  const pmidCol = col('pubmed_id');
  const doiCol  = col('doi');

  const miss = [];
  if (accCol  < 0) miss.push('accession code');
  if (pmcCol  < 0) miss.push('pmc_id');
  if (pmidCol < 0) miss.push('pubmed_id');
  if (doiCol  < 0) miss.push('doi');
  if (miss.length) {
    SpreadsheetApp.getUi().alert('Missing: ' + miss.join(', ') + '\n\n' + headers.join(' | '));
    return;
  }

  const nonEmpty = v => String(v).trim() !== '';
  const isSentinel = v => /^accession_not_found$/i.test(String(v).trim());

  // groups: each tracks total + how many have each identifier
  const mk = () => ({ n: 0, pmc: 0, pmid: 0, pmidNoPmc: 0, doiOnly: 0, none: 0 });
  const g = { sentinel: mk(), blank: mk(), filled: mk() };

  for (let i = 1; i < data.length; i++) {
    const row = data[i];
    if (row.every(c => String(c).trim() === '')) continue;

    const raw = String(row[accCol]).trim();
    let bucket;
    if (raw === '') bucket = g.blank;
    else if (isSentinel(raw)) bucket = g.sentinel;
    else bucket = g.filled;

    bucket.n++;
    const hasPmc  = nonEmpty(row[pmcCol]);
    const hasPmid = nonEmpty(row[pmidCol]);
    const hasDoi  = nonEmpty(row[doiCol]);

    if (hasPmc)  bucket.pmc++;
    if (hasPmid) bucket.pmid++;
    if (hasPmid && !hasPmc) bucket.pmidNoPmc++;
    if (hasDoi && !hasPmc && !hasPmid) bucket.doiOnly++;
    if (!hasPmc && !hasPmid && !hasDoi) bucket.none++;
  }

  const fmt = (label, x) => {
    const p = n => x.n ? (100 * n / x.n).toFixed(0) + '%' : '0%';
    return label + ' (' + x.n + ' rows)\n' +
      '   has pmc_id:            ' + x.pmc + '  (' + p(x.pmc) + ')\n' +
      '   has pubmed_id:         ' + x.pmid + '  (' + p(x.pmid) + ')\n' +
      '   pubmed but no pmc:     ' + x.pmidNoPmc + '  (' + p(x.pmidNoPmc) + ')\n' +
      '   doi only (no pmc/pmid):' + x.doiOnly + '  (' + p(x.doiOnly) + ')\n' +
      '   NO identifier at all:  ' + x.none + '  (' + p(x.none) + ')\n';
  };

  const report =
    'IDENTIFIER REACHABILITY\n\n' +
    fmt('SENTINEL (old pipeline failed)', g.sentinel) + '\n' +
    fmt('BLANK (never attempted)', g.blank) + '\n' +
    fmt('FILLED (has accession, context)', g.filled);

  Logger.log(report);
  SpreadsheetApp.getUi().alert(report);
}