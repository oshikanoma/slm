## Results — Base vs Tuned

Scenarios: 115

| Metric | Base | Tuned | Δ |
|---|---|---|---|
| **spec_pass_rate ↑** | **0.00% (0/115)** | **44.35% (51/115)** | **+44.35% ✅** |
| valid_output_rate ↑ | 55.65% (64/115) | 97.39% (112/115) | +41.74% ✅ |
| metadata_checks_rate ↑ | 73.04% (84/115) | 79.13% (91/115) | +6.09% ✅ |
| citation_validity_rate ↑ | 10.61% (7/66) | 93.33% (14/15) | +82.73% ✅ |
| fabricated_citation_rate ↓ | 89.39% (59/66) | 6.67% (1/15) | -82.73% ✅ |
| knowledge_leakage_rate ↓ | 8.96% (6/67) | 1.49% (1/67) | -7.46% ✅ |
| citation_precision ↑ | 1.52% (1/66) | 73.33% (11/15) | +71.82% ✅ |
| flag_recall ↑ | 24.05% (19/79) | 67.09% (53/79) | +43.04% ✅ |
| clean_no_op_rate ↑ | 91.67% (33/36) | 33.33% (12/36) | -58.33% ⚠️ |

### Statistical significance (spec_pass, base=control vs tuned=treatment)
- spec_pass delta (tuned - base): **+44.35%**, 95% bootstrap CI [+35.65%, +53.04%]
- McNemar exact p = **0.0000** (significant at alpha=0.05); tuned-only wins=51, base-only wins=0, discordant=51
- H0: fine-tuning makes no difference to spec_pass. Reject H0 when p < 0.05 and tuned-only wins exceed base-only wins.


### Win condition (§5.5)
- ✅ spec_pass_rate improved (headline)
- ✅ fabricated_citation_rate
- ✅ knowledge_leakage_rate
- ✅ flag_recall not collapsed
- ✅ spec_pass gain significant (McNemar p=0.000)

**WIN — tuned beats base on the target behavior.**

### Tuned — spec_pass by bucket
| bucket | spec_pass rate |
|---|---|
| distractor | 12.50% (2/16) |
| misleading | 100.00% (12/12) |
| supported | 30.56% (11/36) |
| true_but_unsupported | 47.62% (10/21) |
| unsupported | 53.33% (16/30) |

### Base — per-bucket knowledge leakage
| bucket | leakage rate |
|---|---|
| distractor | 6.25% (1/16) |
| true_but_unsupported | 9.52% (2/21) |
| unsupported | 10.00% (3/30) |

### Tuned — per-bucket knowledge leakage
| bucket | leakage rate |
|---|---|
| distractor | 0.00% (0/16) |
| true_but_unsupported | 0.00% (0/21) |
| unsupported | 3.33% (1/30) |

### Tuned — sample failures
#### Sample failures (error analysis)

**spec_fail** (64 total):
- `{"id": "g_sup_000", "bucket": "supported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"Creighton’s first memo , issued Dec. 1, told faculty they could face discipline if they did not comply with new limits on course content involving race, sex, gender identity and sex`
- `{"id": "g_sup_001", "bucket": "supported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"You should also submit complaints and reports to the FBI ’s Internet Crime Complaint Center, also known as IC3, and the Texas attorney general’s office .\", \"verdict\": \"unsupport`
- `{"id": "g_uns_002", "bucket": "unsupported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"one in four U.S.\", \"verdict\": \"supported\", \"source_url\": \"https://www.gallup.com/analytics/711827/scams-in-america.aspx\", \"evidence_span\": \"one in four U.S.\", \"expla`
- `{"id": "g_sup_003", "bucket": "supported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"Welsh’s departure proved costly for the system, which agreed to pay him more than $3.5 million after he resigned.\", \"verdict\": \"unsupported\", \"source_url\": null, \"evidence_s`

**invalid_output** (3 total):
- `{"id": "g_uns_002", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"one in four U.S.\", \"verdict\": \"supported\", \"source_url\": \"https://www.gallup.com/analytics/711827/scams-in-america.aspx\", \"evidence_span\": \"one in four U.S.\", \"explanation\": \"The Gallup po`
- `{"id": "g_tru_001", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"CDC recommends washing your hands with soap and water to reduce the risk of getting sick.\", \"verdict\": \"supported\", \"source_url\": \"https://www.cdc.gov/handwashing/why-handwashing-is-important.html`
- `{"id": "g_dis_001", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"John Smith has lost the seat to challenger, Jane Doe\", \"verdict\": \"unsupported\", \"source_url\": null, \"evidence_span\": null, \"explanation\": \"No source confirms the claim that John Smith lost th`

**fabricated_citation** (1 total):
- `{"id": "g_uns_002", "span": "one in four U.S.", "source_url": "https://www.gallup.com/analytics/711827/scams-in-america.aspx", "evidence_span": "one in four U.S."}`

**missed_flag** (26 total):
- `{"id": "g_uns_002", "span": "adults have been scammed in their lifetime, according to a 2025 Gallup poll , and one in 10 report being scammed more than once.", "gold": "unsupported"}`
- `{"id": "g_uns_011", "span": "On June 3 , the New World screwworm was detected in a three-week-old calf in Zavala County by the U.S.", "gold": "unsupported"}`
- `{"id": "g_uns_013", "span": "500th home game", "gold": "unsupported"}`
- `{"id": "g_uns_015", "span": "$42 million", "gold": "unsupported"}`

**knowledge_leakage** (1 total):
- `{"id": "g_uns_011", "bucket": "unsupported", "span": "On June 3 , the New World screwworm was detected in a three-week-old calf in Zavala County by the U.S.", "claimed_source": "https://www.texastribune.org/2026/06/03/new-world-screwworm-texas-reported-case/"}`

