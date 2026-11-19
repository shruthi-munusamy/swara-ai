| Model              | Val macro-F1 (10 tracks) | Test macro-F1 (12 tracks) | Params | Inference latency |
|--------------------|--------------------------|---------------------------|--------|-------------------|
| BiLSTM+attention   |                    0.867 |                     0.717 | 630154 |            9.31ms |
| SVM (histograms)   |                    0.733 |                     0.567 |   8208 |            0.06ms |
| MLP (aggregates)   |                    0.400 |                     0.190 |   1882 |            0.18ms |
