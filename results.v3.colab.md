## Results — Base vs Tuned

Scenarios: 115

| Metric | Base | Tuned | Δ |
|---|---|---|---|
| **spec_pass_rate ↑** | **0.00% (0/115)** | **47.83% (55/115)** | **+47.83% ✅** |
| valid_output_rate ↑ | 55.65% (64/115) | 97.39% (112/115) | +41.74% ✅ |
| metadata_checks_rate ↑ | 73.04% (84/115) | 77.39% (89/115) | +4.35% ✅ |
| citation_validity_rate ↑ | 10.61% (7/66) | 90.48% (19/21) | +79.87% ✅ |
| fabricated_citation_rate ↓ | 89.39% (59/66) | 9.52% (2/21) | -79.87% ✅ |
| knowledge_leakage_rate ↓ | 8.96% (6/67) | 1.49% (1/67) | -7.46% ✅ |
| citation_precision ↑ | 1.52% (1/66) | 66.67% (14/21) | +65.15% ✅ |
| flag_recall ↑ | 24.05% (19/79) | 69.62% (55/79) | +45.57% ✅ |
| clean_no_op_rate ↑ | 91.67% (33/36) | 52.78% (19/36) | -38.89% ⚠️ |

### Statistical significance (spec_pass, base=control vs tuned=treatment)
- spec_pass delta (tuned - base): **+47.83%**, 95% bootstrap CI [+39.13%, +56.52%]
- McNemar exact p = **0.0000** (significant at alpha=0.05); tuned-only wins=55, base-only wins=0, discordant=55
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
| distractor | 18.75% (3/16) |
| misleading | 100.00% (12/12) |
| supported | 38.89% (14/36) |
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

**spec_fail** (60 total):
- `{"id": "g_sup_001", "bucket": "supported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"You should also submit complaints and reports to the FBI ’s Internet Crime Complaint Center, also known as IC3, and the Texas attorney general’s office .\", \"verdict\": \"unsupport`
- `{"id": "g_uns_002", "bucket": "unsupported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"one in four U.S.\", \"verdict\": \"supported\", \"source_url\": \"https://www.gallup.com/analytics/711827/scams-in-america.aspx\", \"evidence_span\": \"scams affected millions of `
- `{"id": "g_sup_004", "bucket": "supported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"Trump and his administration have routinely said they are targeting immigrants who have a criminal history, but the federal government’s data shows that a majority of the people ICE`
- `{"id": "g_sup_005", "bucket": "supported", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"In Texas last year, an agent shot and killed a man on South Padre Island .\", \"verdict\": \"unsupported\", \"source_url\": null, \"evidence_span\": null, \"explanation\": \"No prov`

**missed_flag** (24 total):
- `{"id": "g_uns_002", "span": "adults have been scammed in their lifetime, according to a 2025 Gallup poll , and one in 10 report being scammed more than once.", "gold": "unsupported"}`
- `{"id": "g_uns_011", "span": "On June 3 , the New World screwworm was detected in a three-week-old calf in Zavala County by the U.S.", "gold": "unsupported"}`
- `{"id": "g_uns_013", "span": "500th home game", "gold": "unsupported"}`
- `{"id": "g_uns_015", "span": "$42 million", "gold": "unsupported"}`

**invalid_output** (3 total):
- `{"id": "g_sup_006", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"“Free and fair elections are a cornerstone of a thriving republic, and with the authority granted to my office by the Legislature, we will stop at nothing to uncover and stop any illegal voting activity,”`
- `{"id": "g_sup_020", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"The project could represent $1 billion in private capital investment , support 500 construction jobs and 30 full-time positions once completed, according to a fact sheet created by the city of Lufkin.\", `
- `{"id": "g_dis_001", "raw": "{\"clean\": false, \"verdicts\": [{\"type\": \"claim\", \"span\": \"John Smith has lost the seat to challenger, Jane Doe\", \"verdict\": \"unsupported\", \"source_url\": null, \"evidence_span\": null, \"explanation\": \"No source confirms the claim that John Smith lost th`

**fabricated_citation** (2 total):
- `{"id": "g_sup_006", "span": "“Free and fair elections are a cornerstone of a thriving republic, and with the authority granted to my office by the Legislature, we will stop at nothing to uncover and stop any illegal voting activity,” Paxton said in a February news release announcing the tip line.", `
- `{"id": "g_sup_020", "span": "The project could represent $1 billion in private capital investment , support 500 construction jobs and 30 full-time positions once completed, according to a fact sheet created by the city of Lufkin.", "source_url": "https://www.cityoflufkin.com/_T2_R653.php", "evidence`

**knowledge_leakage** (1 total):
- `{"id": "g_uns_011", "bucket": "unsupported", "span": "On June 3 , the New World screwworm was detected in a three-week-old calf in Zavala County by the U.S.", "claimed_source": "https://www.texastribune.org/2026/06/03/new-world-screwworm-texas-reported-case/"}`

