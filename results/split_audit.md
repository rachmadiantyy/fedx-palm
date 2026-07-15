# Split audit

Splits dir: `data\splits_v2`

| Split | Images | Boxes | Unique bunches | Img/bunch (min/med/max) | Abnormal | Empty Bunch | Overripe | Ripe | Underripe | Unripe |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 8937 | 30688 | 72 | 56/100.0/1075 | 5722 | 1357 | 5485 | 6534 | 5576 | 6014 |
| val | 826 | 3186 | 8 | 89/103.5/118 | 727 | 94 | 518 | 518 | 378 | 951 |
| test | 1051 | 3822 | 11 | 69/99/115 | 408 | 761 | 797 | 490 | 679 | 687 |

- Bunch_ids in >1 split: **0** (must be 0)
- Source groups (alias-aware) in >1 split: **0** (must be 0)
- Exact-duplicate files across splits (MD5): **0** (must be 0)
- Raw dHash candidates across splits (Hamming<=3): **13** (informational -- gated via review below)
    - confirmed cross-split duplicates: **0** (must be 0)
    - reviewed false positives (different_source): 13 (do not gate)
    - uncertain / unreviewed: **0** (must be 0 -- review via scripts/13 + data/source_alias_review.csv)
- **Leakage audit: PASSED**

## Sample filename -> bunch_id mappings (parser sanity check)

**train:**
- `framesawit24-30-_png_jpg.rf.bf07988ab577bf58e0453e2da2c87eef.jpg` -> `framesawit24`
- `framesawit31-1-_png_jpg.rf.ae40b6ae3c25c8db72dda79f033bdc43.jpg` -> `framesawit31`
- `frame10kombinasi-286-_png_jpg.rf.3cd0f275b8834fe622ae84efd3131c8b.jpg` -> `frame10kombinasi`
- `framementah21-25-_png_jpg.rf.54be3bccdbc0a213f89b33f9bec6e294.jpg` -> `framementah21`
- `frameterlalumasak12-3-_png_jpg.rf.23d1078933470fbae14b2657cd5bb810.jpg` -> `frameterlalumasak12`
- `framesawit7-39-_png_jpg.rf.6747e99ff98589e92537962129bdb986.jpg` -> `framesawit7`
- `framesawit28-24-_png_jpg.rf.63d7c3e32079458b95e723408700fca6.jpg` -> `framesawit28`
- `framesawit1-34-_png_jpg.rf.c93e96c06b37567002f8bb2e992d95c3.jpg` -> `framesawit1`
- `framesawit5-22-_png_jpg.rf.ed8711be9b18222173c6de8a6533e9d2.jpg` -> `framesawit5`
- `framesawit20-31-_png_jpg.rf.b039957064e9c5c11bc1cbf639333398.jpg` -> `framesawit20`
- `framementah13-18-_png_jpg.rf.efd914f7e2d5dcd2e0bf1e0dafcbb1d2.jpg` -> `framementah13`
- `frameterlalumasak11-24-_png_jpg.rf.feda6da97b8b91f06a012caaa2b5ac3e.jpg` -> `frameterlalumasak11`

**val:**
- `framementah4-26-_png_jpg.rf.edf9949d4f108ddf8b2b751010c9d105.jpg` -> `framementah4`
- `framesawit14-37-_png_jpg.rf.53dd28ea610bb7041db83046bdd31d1d.jpg` -> `framesawit14`
- `framementah4-26-_png_jpg.rf.efb1441a480be2074bd10ee9e9953777.jpg` -> `framementah4`
- `frameterlalumasak13-3-_png_jpg.rf.3b3f38cab36a12e46436e256d9d7e9f4.jpg` -> `frameterlalumasak13`
- `framementah4-10-_png_jpg.rf.dab261bd948fc59c76a6e8a67b8effe1.jpg` -> `framementah4`
- `framesawit8-20-_png_jpg.rf.2ca7b72b3cea7e3eab150eef66e4c5e9.jpg` -> `framesawit8`
- `frameterlalumasak13-7-_png_jpg.rf.4ba4ae16c7e5fab9adec0e5bb9830ea2.jpg` -> `frameterlalumasak13`
- `framesawit14-26-_png_jpg.rf.09d194dd880906b53a3f0e9ea99c871a.jpg` -> `framesawit14`
- `framesawit6-22-_png_jpg.rf.b1eceb249a73253d22eedbaf68b8bf76.jpg` -> `framesawit6`
- `frameterlalumasak13-10-_png_jpg.rf.fb27410ac5e0368f01d1e730244d6cb4.jpg` -> `frameterlalumasak13`
- `framesawit8-13-_png_jpg.rf.7dc3562bdab55590490760125eb2bf05.jpg` -> `framesawit8`
- `framementah4-29-_png_jpg.rf.094c7fe8e0edb3c2e7442f79ac9cf95e.jpg` -> `framementah4`

**test:**
- `framesawit26-40-_png_jpg.rf.5e34066aa010feff756fa9846d488644.jpg` -> `framesawit26`
- `frame12-9-_png_jpg.rf.924aeb5d81be522fe109c5733fbc7b7e.jpg` -> `frame12`
- `frame12-26-_png_jpg.rf.e704d5fa17bbe032ed36434fb7b9b5a0.jpg` -> `frame12`
- `framesawit3-18-_png_jpg.rf.6c517195537eea4e1c762d9a242902c1.jpg` -> `framesawit3`
- `framesawit42-21-_png_jpg.rf.bffc2cae9fd14200d82531d3a8a84429.jpg` -> `framesawit42`
- `framekurangmasak10-1-_png_jpg.rf.7c774fd765159cc105248ea21146f999.jpg` -> `framekurangmasak10`
- `framesawit3-35-_png_jpg.rf.b048f437ec2456da93bd4866441f22f4.jpg` -> `framesawit3`
- `framesawit39-28-_png_jpg.rf.d2e6fbcb6cf9b2ddb57660b12641939c.jpg` -> `framesawit39`
- `framesawit26-8-_png_jpg.rf.4aca7dfd31b0a6cd0f02bdcf995f3058.jpg` -> `framesawit26`
- `framementah15-31-_png_jpg.rf.b0d1e2498704e514cd10710c1feb3e61.jpg` -> `framementah15`
- `framesawit42-25-_png_jpg.rf.3ab3305ebf2c986f8ec1fdaaa875220f.jpg` -> `framesawit42`
- `framesawit39-34-_png_jpg.rf.e611432b7405b6fdb8afe6ffc3ddfb24.jpg` -> `framesawit39`

