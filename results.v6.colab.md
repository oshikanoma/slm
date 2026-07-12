## Results — Base vs Tuned

Scenarios: 175

| Metric | Base | Tuned | Δ |
|---|---|---|---|
| **spec_pass_rate ↑** | **0.57% (1/175)** | **84.00% (147/175)** | **+83.43% ✅** |
| valid_output_rate ↑ | 72.00% (126/175) | 97.14% (170/175) | +25.14% ✅ |
| metadata_checks_rate ↑ | 95.43% (167/175) | 99.43% (174/175) | +4.00% ✅ |
| citation_validity_rate ↑ | 54.55% (12/22) | 94.44% (34/36) | +39.90% ✅ |
| fabricated_citation_rate ↓ | 45.45% (10/22) | 5.56% (2/36) | -39.90% ✅ |
| knowledge_leakage_rate ↓ | 2.99% (2/67) | 7.46% (5/67) | +4.48% ⚠️ |
| citation_precision ↑ | 18.18% (4/22) | 55.56% (20/36) | +37.37% ✅ |
| flag_recall ↑ | 9.35% (13/139) | 85.61% (119/139) | +76.26% ✅ |
| clean_no_op_rate ↑ | 91.67% (33/36) | 83.33% (30/36) | -8.33% ⚠️ |

### Statistical significance (spec_pass, base=control vs tuned=treatment)
- spec_pass delta (tuned - base): **+83.43%**, 95% bootstrap CI [+77.71%, +89.14%]
- McNemar exact p = **0.0000** (significant at alpha=0.05); tuned-only wins=147, base-only wins=1, discordant=148
- H0: fine-tuning makes no difference to spec_pass. Reject H0 when p < 0.05 and tuned-only wins exceed base-only wins.


### Win condition (§5.5)
- ✅ spec_pass_rate improved (headline)
- ✅ fabricated_citation_rate
- ❌ knowledge_leakage_rate
- ✅ flag_recall not collapsed
- ✅ spec_pass gain significant (McNemar p=0.000)

**NOT YET — see failed checks above.**

### Tuned — spec_pass by bucket
| bucket | spec_pass rate |
|---|---|
| ap_style | 100.00% (60/60) |
| distractor | 87.50% (14/16) |
| misleading | 100.00% (12/12) |
| supported | 52.78% (19/36) |
| true_but_unsupported | 85.71% (18/21) |
| unsupported | 80.00% (24/30) |

### Base — per-bucket knowledge leakage
| bucket | leakage rate |
|---|---|
| distractor | 6.25% (1/16) |
| true_but_unsupported | 0.00% (0/21) |
| unsupported | 3.33% (1/30) |

### Tuned — per-bucket knowledge leakage
| bucket | leakage rate |
|---|---|
| distractor | 6.25% (1/16) |
| true_but_unsupported | 0.00% (0/21) |
| unsupported | 13.33% (4/30) |

### Tuned — sample failures
#### Sample failures (error analysis)

**spec_fail** (28 total):
- `{"id": "g_uns_001", "bucket": "unsupported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"A second memo , issued April 9, went further, ordering the phase-out of academic programs centered on sexual orientation and gender identity and requiring professors in core and l`
- `{"id": "g_uns_002", "bucket": "unsupported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"one in four U.S. adults have been scammed in their lifetime, according to a 2025 Gallup poll , and one in 10 report being scammed more than once.\", \"verdict\": \"supported\", \"`
- `{"id": "g_sup_004", "bucket": "supported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"Trump and his administration have routinely said they are targeting immigrants who have a criminal history, but the federal government’s data shows that a majority of the people ICE`
- `{"id": "g_sup_006", "bucket": "supported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"“Free and fair elections are a cornerstone of a thriving republic, and with the authority granted to my office by the Legislature, we will stop at nothing to uncover and stop any il`

**knowledge_leakage** (5 total):
- `{"id": "g_uns_001", "bucket": "unsupported", "span": "A second memo , issued April 9, went further, ordering the phase-out of academic programs centered on sexual orientation and gender identity and requiring professors in core and lower-level undergraduate courses to use alternate materials if read`
- `{"id": "g_uns_002", "bucket": "unsupported", "span": "adults have been scammed in their lifetime, according to a 2025 Gallup poll , and one in 10 report being scammed more than once.", "claimed_source": "https://www.gallup.com/analytics/711827/scams-in-america.aspx"}`
- `{"id": "g_uns_009", "bucket": "unsupported", "span": "Hinojosa has embraced populist policy in her challenge to Abbott, whom she has accused of working on behalf of GOP megadonors at the expense of everyday Texans.", "claimed_source": "https://www.texastribune.org/2026/06/26/texas-democratic-convent`
- `{"id": "g_uns_011", "bucket": "unsupported", "span": "On June 3 , the New World screwworm was detected in a three-week-old calf in Zavala County by the U.S.", "claimed_source": "https://www.texastribune.org/2026/06/03/new-world-screwworm-texas-reported-case/"}`

**missed_flag** (20 total):
- `{"id": "g_uns_001", "span": "A second memo , issued April 9, went further, ordering the phase-out of academic programs centered on sexual orientation and gender identity and requiring professors in core and lower-level undergraduate courses to use alternate materials if readings, assignments or lect`
- `{"id": "g_uns_002", "span": "adults have been scammed in their lifetime, according to a 2025 Gallup poll , and one in 10 report being scammed more than once.", "gold": "unsupported"}`
- `{"id": "g_uns_009", "span": "Hinojosa has embraced populist policy in her challenge to Abbott, whom she has accused of working on behalf of GOP megadonors at the expense of everyday Texans.", "gold": "unsupported"}`
- `{"id": "g_uns_011", "span": "On June 3 , the New World screwworm was detected in a three-week-old calf in Zavala County by the U.S.", "gold": "unsupported"}`

**invalid_output** (5 total):
- `{"id": "g_sup_006", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"“Free and fair elections are a cornerstone of a thriving republic, and with the authority granted to my office by the Legislature, we will stop at nothing to uncover and stop any illegal voting activity,”`
- `{"id": "g_sup_011", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"In fourth grade, students would encounter Luke 14:7-11 , a New Testament passage where Jesus says: \\\"All those who lift themselves up will be made humble. And those who make themselves humble will be li`
- `{"id": "g_sup_020", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"The project could represent $1 billion in private capital investment , support 500 construction jobs and 30 full-time positions once completed, according to a fact sheet created by the city of Lufkin.\", `
- `{"id": "g_tru_001", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"CDC recommends washing your hands with soap and water to reduce the risk of getting sick.\", \"verdict\": \"supported\", \"source_url\": \"https://www.cdc.gov/handwashing/why-handwashing-is-important.html`

**fabricated_citation** (2 total):
- `{"id": "g_sup_006", "span": "“Free and fair elections are a cornerstone of a thriving republic, and with the authority granted to my office by the Legislature, we will stop at nothing to uncover and stop any illegal voting activity,” Paxton said in a February news release announcing the tip line.", `
- `{"id": "g_sup_020", "span": "The project could represent $1 billion in private capital investment , support 500 construction jobs and 30 full-time positions once completed, according to a fact sheet created by the city of Lufkin.", "source_url": "https://www.cityoflufkin.com/_T2_R653.php", "evidence`

