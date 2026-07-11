## Results — Base vs Tuned

Scenarios: 135

| Metric | Base | Tuned | Δ |
|---|---|---|---|
| **spec_pass_rate ↑** | **0.74% (1/135)** | **54.81% (74/135)** | **+54.07% ✅** |
| valid_output_rate ↑ | 68.15% (92/135) | 97.78% (132/135) | +29.63% ✅ |
| metadata_checks_rate ↑ | 67.41% (91/135) | 80.00% (108/135) | +12.59% ✅ |
| citation_validity_rate ↑ | 54.55% (12/22) | 92.31% (24/26) | +37.76% ✅ |
| fabricated_citation_rate ↓ | 45.45% (10/22) | 7.69% (2/26) | -37.76% ✅ |
| knowledge_leakage_rate ↓ | 1.49% (1/67) | 4.48% (3/67) | +2.99% ⚠️ |
| citation_precision ↑ | 13.64% (3/22) | 57.69% (15/26) | +44.06% ✅ |
| flag_recall ↑ | 13.13% (13/99) | 76.77% (76/99) | +63.64% ✅ |
| clean_no_op_rate ↑ | 91.67% (33/36) | 63.89% (23/36) | -27.78% ⚠️ |

### Statistical significance (spec_pass, base=control vs tuned=treatment)
- spec_pass delta (tuned - base): **+54.07%**, 95% bootstrap CI [+45.19%, +62.22%]
- McNemar exact p = **0.0000** (significant at alpha=0.05); tuned-only wins=73, base-only wins=0, discordant=73
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
| ap_style | 100.00% (20/20) |
| distractor | 12.50% (2/16) |
| misleading | 100.00% (12/12) |
| supported | 41.67% (15/36) |
| true_but_unsupported | 47.62% (10/21) |
| unsupported | 50.00% (15/30) |

### Base — per-bucket knowledge leakage
| bucket | leakage rate |
|---|---|
| distractor | 0.00% (0/16) |
| true_but_unsupported | 0.00% (0/21) |
| unsupported | 3.33% (1/30) |

### Tuned — per-bucket knowledge leakage
| bucket | leakage rate |
|---|---|
| distractor | 0.00% (0/16) |
| true_but_unsupported | 0.00% (0/21) |
| unsupported | 10.00% (3/30) |

### Tuned — sample failures
#### Sample failures (error analysis)

**spec_fail** (61 total):
- `{"id": "g_uns_001", "bucket": "unsupported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"A second memo , issued April 9, went further, ordering the phase-out of academic programs centered on sexual orientation and gender identity and requiring professors in core and l`
- `{"id": "g_sup_001", "bucket": "supported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"You should also submit complaints and reports to the FBI ’s Internet Crime Complaint Center, also known as IC3, and the Texas attorney general’s office .\", \"verdict\": \"unsupport`
- `{"id": "g_uns_002", "bucket": "unsupported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"one in four U.S. adults have been scammed in their lifetime, according to a 2025 Gallup poll , and one in 10 report being scammed more than once.\", \"verdict\": \"supported\", \"`
- `{"id": "g_sup_004", "bucket": "supported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"Trump and his administration have routinely said they are targeting immigrants who have a criminal history, but the federal government’s data shows that a majority of the people ICE`

**knowledge_leakage** (3 total):
- `{"id": "g_uns_001", "bucket": "unsupported", "span": "A second memo , issued April 9, went further, ordering the phase-out of academic programs centered on sexual orientation and gender identity and requiring professors in core and lower-level undergraduate courses to use alternate materials if read`
- `{"id": "g_uns_002", "bucket": "unsupported", "span": "adults have been scammed in their lifetime, according to a 2025 Gallup poll , and one in 10 report being scammed more than once.", "claimed_source": "https://www.gallup.com/analytics/711827/scams-in-america.aspx"}`
- `{"id": "g_uns_011", "bucket": "unsupported", "span": "On June 3 , the New World screwworm was detected in a three-week-old calf in Zavala County by the U.S.", "claimed_source": "https://www.texastribune.org/2026/06/03/new-world-screwworm-texas-reported-case/"}`

**missed_flag** (23 total):
- `{"id": "g_uns_001", "span": "A second memo , issued April 9, went further, ordering the phase-out of academic programs centered on sexual orientation and gender identity and requiring professors in core and lower-level undergraduate courses to use alternate materials if readings, assignments or lect`
- `{"id": "g_uns_002", "span": "adults have been scammed in their lifetime, according to a 2025 Gallup poll , and one in 10 report being scammed more than once.", "gold": "unsupported"}`
- `{"id": "g_uns_011", "span": "On June 3 , the New World screwworm was detected in a three-week-old calf in Zavala County by the U.S.", "gold": "unsupported"}`
- `{"id": "g_uns_013", "span": "500th home game", "gold": "unsupported"}`

**invalid_output** (3 total):
- `{"id": "g_sup_006", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"“Free and fair elections are a cornerstone of a thriving republic, and with the authority granted to my office by the Legislature, we will stop at nothing to uncover and stop any illegal voting activity,”`
- `{"id": "g_sup_020", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"The project could represent $1 billion in private capital investment , support 500 construction jobs and 30 full-time positions once completed, according to a fact sheet created by the city of Lufkin.\", `
- `{"id": "g_dis_005", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"The suspect was 'armed and dangerous'\", \"verdict\": \"unsupported\", \"source_url\": null, \"evidence_span\": null, \"explanation\": \"No source confirms the suspect was armed and dangerous.\", \"checke`

**fabricated_citation** (2 total):
- `{"id": "g_sup_006", "span": "“Free and fair elections are a cornerstone of a thriving republic, and with the authority granted to my office by the Legislature, we will stop at nothing to uncover and stop any illegal voting activity,” Paxton said in a February news release announcing the tip line.", `
- `{"id": "g_sup_020", "span": "The project could represent $1 billion in private capital investment , support 500 construction jobs and 30 full-time positions once completed, according to a fact sheet created by the city of Lufkin.", "source_url": "https://www.cityoflufkin.com/_T2_R653.php", "evidence`

