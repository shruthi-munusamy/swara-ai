| Model              | Val macro-F1 | Test macro-F1 | Params | Inference latency |
|--------------------|--------------|---------------|--------|--------------------|
| SVM (histograms)   |        0.733 |         0.567 |   8208 |           0.07ms |
| MLP (aggregates)   |        0.400 |         0.190 |   1882 |           0.06ms |
| BiLSTM+attention   |         n/a* |         0.717 | 630154 |          12.02ms |

*LSTM val macro-F1 (0.595, best epoch 13) comes from the training run log, not recomputed by this script -- see README training section.
