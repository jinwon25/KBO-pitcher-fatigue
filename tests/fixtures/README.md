# 2025 doubleheader regression evidence

`mykbo_doubleheaders_2025.csv` is a manually checked, minimal factual fixture,
not an HTML capture and not an inferred assignment of all 26 legacy rows.
Checked 2026-10-07 against MYKBO's game box scores and source player links:

| Date | Player (source ID) | Game IDs in playing order | Pitch counts |
| --- | --- | --- | --- |
| 2025-05-10 | 장현식 (572) | 13007, 13011 | 14, 16 |
| 2025-05-17 | 박영현 (2302) | 13023, 13028 | 12, 17 |

Each fixture's `GameURL` is its evidence URL. The 13 duplicate legacy keys
(26 participating rows) are tested directly from the unchanged raw snapshot.
Synthetic PBP tests reuse the two dates but do not claim observed pitch events.
MYKBO numeric IDs identify source games; they are not KBO official game IDs.
