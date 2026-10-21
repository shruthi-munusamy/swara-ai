Latency profile -- 10 runs on a 10.4s clip (CREPE model: tiny)

| Stage       | p50 (ms) | p95 (ms) |
|-------------|----------|----------|
| audio_load  |     0.91 |   469.63 |
| crepe       |   871.46 |  3124.13 |
| cents       |     0.06 |     0.25 |
| segment     |     2.34 |     6.42 |
| lstm        |     1.68 |    17.64 |
| total       |   876.65 |  3617.23 |
