#!/usr/bin/env python3
"""TASK 2b -- manual adjudication of every LLM-vs-human metadata disagreement.

All 40 disagreements surfaced by meta_agreement.py were read against the paper
text in all_usable_papers.json and assigned a verdict by hand. Indices are
positions in meta_disagreements.json (seed-99 shuffle, reproducible).

Verdicts:
  llm_right    -- the LLM's value is supported by the paper, the human's is not
  human_right  -- the reverse
  normalisation-- the two strings denote the same thing (alias/format), no real
                  disagreement; counts as agreement once normalised
  schema       -- the human used the "Both" sequencing class, which the LLM's
                  output schema does not contain. Not an LLM factual error; a
                  fixable prompt defect. Scored separately.
  undetermined -- the scraped text is abstract-only and states no method

Writes meta_adjudication.json.
"""
import json, os, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))

# idx -> (verdict, note)
COUNTRY = {
    1:  ("llm_right", "All affiliations Japan (NCNP Tokyo, RIKEN); ethics = NCNP Ethics Committee. "
                      "'United States' occurs ONLY in the journal name (Proc Natl Acad Sci U S A) and "
                      "the reference list. Human recorded the journal's country."),
    4:  ("normalisation", "'Hong Kong SAR, China' vs 'China' -- same country, finer granularity."),
    5:  ("llm_right", "Cohort = Geneva Memory Center, ethics = Geneva Ethics Committee (CCER). "
                      "Human took the FIRST author's affiliation (Leiden, Netherlands)."),
    7:  ("normalisation", "'Türkiye' vs 'Turkey' -- endonym vs exonym."),
    10: ("llm_right", "Samples = DeNoPa cohort, Paracelsus-Elena-Klinik Kassel, ethics Physician's "
                      "Board Hessen, Germany. Human took the lead LAB's country (Luxembourg)."),
    12: ("normalisation", "'PR China' vs 'China'."),
    13: ("llm_right", "SILCODE cohort recruited in Beijing; all study sites Chinese. 'Canada' is the "
                      "LAST author's second affiliation (University of Calgary) only."),
    18: ("normalisation", "'Netherlands, Finland' vs 'Netherlands and Finland' -- separator only."),
    20: ("llm_right", "Recruited at National MS Center Melsbroek + University Hospital Brussels; ethics "
                      "Brussels. 'USA' occurs only in publisher boilerplate ('American Neurological "
                      "Association') and one reference."),
    21: ("llm_right", "Title says 'Chinese Patients'; cohort at Sun Yat-sen Memorial Hospital, Guangzhou. "
                      "'United States' is the LAST author's affiliation (Univ. of Virginia)."),
    23: ("llm_right", "Patients enrolled at Istanbul Medipol University periodontology clinics. Human took "
                      "the first-listed affiliation (ADA Forsyth Institute, Cambridge MA)."),
    24: ("llm_right", "Re-analysis of the Wallen et al. US cohort ('Participant recruitment has been "
                      "described before by Wallen et al.'); authors are at Univ. of Galway, Ireland. "
                      "The LLM tracked the sample origin across a citation."),
    25: ("llm_right", "Trial run at University Hospital Bonn, ethics committee University of Bonn, Germany. "
                      "'UK' is a co-affiliation (Quadram Institute)."),
    31: ("llm_right", "Participants from the Italian INDIA-FBP study, ethics Brescia, Italy. Switzerland is "
                      "a co-author affiliation (Geneva genetic medicine) with no cohort role."),
    32: ("llm_right", "Cohort at University of Malaya, Kuala Lumpur. 'Canada' is the LAST author's "
                      "affiliation (Toronto Western Hospital)."),
    33: ("llm_right", "Fecal samples collected at Shuang Ho Hospital, Taipei Medical University, Taiwan. "
                      "Human took the first-listed affiliation (Hebrew SeniorLife, Boston)."),
    34: ("llm_right", "Title says 'Older Chinese Adults'; 229 adults recruited in Shenzhen, China. "
                      "'Australia' is a co-affiliation (Monash)."),
    35: ("llm_right", "Medical University of Graz, ethics committee Graz. 'USA' occurs ONLY in reagent "
                      "vendor addresses (R&D Systems Minneapolis; SPSS Chicago) and one background "
                      "sentence about other studies."),
    36: ("llm_right", "Same DeNoPa/Kassel cohort as idx 10; ethics Physician's Board Hesse, Germany. "
                      "Human took the lead lab's country (Luxembourg)."),
    37: ("llm_right", "All affiliations West China Hospital, Sichuan University. 'United States' occurs "
                      "once, in publisher boilerplate."),
}

SEQUENCING = {
    0:  ("llm_right", "Method is '16S and 23S rRNA-targeted quantitative reverse transcription (qRT)-PCR' "
                      "-- an RT-qPCR assay, not amplicon sequencing. Human matched the string '16S'."),
    2:  ("undetermined", "Scraped text is 6.0k chars, abstract-only, no method stated. (Same Japanese "
                         "group as idx 16, which used T-RFLP, so the human's 'Other' is plausible.)"),
    3:  ("schema", "Text confirms both 16S amplicon AND shotgun metagenome sequencing; human 'Both' is "
                   "correct and the LLM schema has no 'Both' class."),
    6:  ("schema", "'characterized using 16S and metagenomic shotgun sequencing' -- human 'Both' correct."),
    8:  ("undetermined", "6.4k chars, abstract-only; no sequencing method stated in the scraped text."),
    9:  ("schema", "16S rDNA sequencing plus a later shotgun run -- human 'Both' correct."),
    11: ("schema", "'The 16S rRNA amplicon and shotgun metagenomic sequencing was performed' -- 'Both'."),
    14: ("undetermined", "8.4k chars, abstract-only; no method statement in the scraped text."),
    15: ("schema", "Cases by whole metagenomic sequencing; controls' 16S pulled from the Taiwan Microbiome "
                   "Database -- human 'Both' defensible."),
    16: ("human_right", "Method is T-RFLP fingerprinting, described as 'one of the most well-established "
                        "16S ribosomal RNA-based methods'. T-RFLP is not sequencing; human 'Other' is "
                        "right. The LLM matched the string '16S rRNA-based'."),
    17: ("llm_right", "SYBR-green real-time PCR with 16S rRNA group-specific primers. The paper itself "
                      "contrasts its method with sequencing: 'they used sequencing technique detecting "
                      "more bacteria than Real time PCR applied in the present research'."),
    19: ("schema", "'16S rRNA gene and whole metagenomic sequencing data' -- human 'Both' correct."),
    22: ("schema", "16S rRNA V4 on 89 subjects + whole metagenomic shotgun on 48 -- 'Both' correct."),
    26: ("llm_right", "'bacterial abundance was quantified by qPCR using 16S rRNA group-specific primers "
                      "... SYBR Green qPCR in a Rotor gene 3000'. No sequencing performed. Human matched "
                      "the string '16S'."),
    27: ("llm_right", "'the gut microbiota of 19 ET patients and 21 HCs were analysed with metagenomics "
                      "approach'. No 16S anywhere in the text."),
    28: ("schema", "16S and 18S amplicon sequencing plus metagenomic shotgun of selected samples."),
    29: ("schema", "'Fecal 16S rRNA and shotgun metagenomic sequencing ... were performed' -- 'Both'."),
    30: ("llm_right", "The study's own method is metagenome sequencing. The 16S mentions are (a) a CITED "
                      "prior CADASIL study, (b) a qPCR normaliser, (c) colony Sanger for isolate ID -- "
                      "none is community 16S profiling. Human 'Both' over-calls."),
    38: ("undetermined", "5.7k chars, abstract-only; 'metagenome' appears only inside a feature name "
                         "('gut_metagenome_g_Faecalibacterium'), which is not method evidence."),
    39: ("llm_right", "Only 'Metagenomic shotgun sequencing was performed on DNA extracted from stool'. "
                      "No 16S in the scraped text; human 'Both' unsupported. (Text is abstract-only, so "
                      "this is the weakest of the llm_right calls.)"),
}


def boot_ci(k, n, seed=0, iters=20000):
    """Bootstrap CI on a rate k/n, resampling PAPERS."""
    if n == 0:
        return [0.0, 0.0]
    flags = [1.0] * k + [0.0] * (n - k)
    rng = random.Random(seed)
    b = sorted(st.mean([flags[rng.randrange(n)] for _ in range(n)]) for _ in range(iters))
    return [round(b[int(.025 * iters)], 4), round(b[int(.975 * iters)], 4)]


def tally(d, name, n_compared):
    c = {}
    for v, _ in d.values():
        c[v] = c.get(v, 0) + 1
    print(f"\n=== {name} ({len(d)} disagreements of {n_compared} compared) ===")
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"   {k:14s} {v:3d}")
    return c


def main():
    dis = json.load(open(os.path.join(HERE, "meta_disagreements.json")))
    agree = json.load(open(os.path.join(HERE, "meta_agreement.json")))
    out = {}

    for field, table, n_comp in [
        ("country", COUNTRY, agree["country"]["n_both_present"]),
        ("sequencing", SEQUENCING, agree["sequencing"]["n_both_present"]),
    ]:
        # sanity: indices must line up with the field
        for i in table:
            assert dis[i]["field"] == field, (field, i, dis[i]["field"])
        c = tally(table, field, n_comp)
        n_norm = c.get("normalisation", 0)
        n_schema = c.get("schema", 0)
        n_und = c.get("undetermined", 0)
        n_llm = c.get("llm_right", 0)
        n_hum = c.get("human_right", 0)

        # adjudicable = papers where a factual verdict was reachable
        adjudicable = n_comp - n_norm - n_schema - n_und
        rec = {
            "n_compared": n_comp,
            "n_disagreements": len(table),
            "counts": c,
            "true_agreement_after_normalisation": round((n_comp - (len(table) - n_norm)) / n_comp, 4),
            "n_adjudicable": adjudicable,
            "human_error_rate": round(n_llm / adjudicable, 4),
            "human_error_ci95": boot_ci(n_llm, adjudicable, seed=hash(field) % 999),
            "llm_error_rate": round(n_hum / adjudicable, 4),
            "llm_error_ci95": boot_ci(n_hum, adjudicable, seed=(hash(field) + 7) % 999),
            "verdicts": {str(i): {"title": dis[i]["title"], "llm": dis[i]["llm"],
                                  "human": dis[i]["human"], "verdict": v, "note": note}
                         for i, (v, note) in table.items()},
        }
        out[field] = rec
        print(f"   agreement after normalisation : {rec['true_agreement_after_normalisation']:.4f}")
        print(f"   adjudicable papers            : {adjudicable}")
        print(f"   HUMAN wrong: {n_llm}/{adjudicable} = {rec['human_error_rate']:.4f} "
              f"CI95 {rec['human_error_ci95']}")
        print(f"   LLM   wrong: {n_hum}/{adjudicable} = {rec['llm_error_rate']:.4f} "
              f"CI95 {rec['llm_error_ci95']}")

    # combined
    tot_adj = out["country"]["n_adjudicable"] + out["sequencing"]["n_adjudicable"]
    tot_h = out["country"]["counts"].get("llm_right", 0) + out["sequencing"]["counts"].get("llm_right", 0)
    tot_l = out["country"]["counts"].get("human_right", 0) + out["sequencing"]["counts"].get("human_right", 0)
    out["combined"] = {
        "n_adjudicable": tot_adj,
        "human_error_rate": round(tot_h / tot_adj, 4), "human_errors": tot_h,
        "human_error_ci95": boot_ci(tot_h, tot_adj, seed=3),
        "llm_error_rate": round(tot_l / tot_adj, 4), "llm_errors": tot_l,
        "llm_error_ci95": boot_ci(tot_l, tot_adj, seed=4),
    }
    print(f"\n=== COMBINED (both fields, {tot_adj} adjudicable paper-fields) ===")
    print(f"   human wrong {tot_h} = {out['combined']['human_error_rate']:.4f} "
          f"CI95 {out['combined']['human_error_ci95']}")
    print(f"   llm   wrong {tot_l} = {out['combined']['llm_error_rate']:.4f} "
          f"CI95 {out['combined']['llm_error_ci95']}")

    # Exact binomial-ish sign test: of the adjudicated disagreements, how often
    # was the LLM the correct one? Null = 50/50 coin flip per disagreement.
    n = tot_h + tot_l
    from math import comb
    p = sum(comb(n, k) for k in range(tot_h, n + 1)) / 2 ** n
    out["combined"]["sign_test_p_one_sided"] = round(p, 6)
    out["combined"]["sign_test_note"] = (
        f"Of {n} adjudicated disagreements the LLM was right in {tot_h}. "
        f"Exact binomial vs a 50/50 null, one-sided p={p:.2e}. One observation per PAPER.")
    print(f"   sign test: LLM right in {tot_h}/{n} adjudicated disagreements, p={p:.2e}")

    json.dump(out, open(os.path.join(HERE, "meta_adjudication.json"), "w"), indent=2)
    print("\nwrote meta_adjudication.json")


if __name__ == "__main__":
    main()
