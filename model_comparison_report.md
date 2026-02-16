# E. coli Gene Function Classification Report

## 1. Objective
To classify E. coli protein sequences into functional categories (e.g., 'DNA-binding protein', 'transporter') using multiple model architectures.

## 2. Models Compared
1. **ProteinCausalLM (Transformer)**: A causal language model adapted for classification.
   - Architecture: 6 layers, 8 heads, d_model=256
   - Mechanism: Mean pooling of encoder outputs + Linear Classifier.
   
2. **ProteinBiLSTM**: A Bidirectional LSTM model.
   - Architecture: 6 layers, hidden_size=256
   - Mechanism: Mean pooling of hidden states + Linear Classifier.

## 3. Results Summary

| Model       | Type          | Fold               |   Val Loss |   Val Accuracy | Convergence     |
|:------------|:--------------|:-------------------|-----------:|---------------:|:----------------|
| transformer | Deep Learning | transformer_fold_0 |          0 |              0 | See TensorBoard |
| bilstm      | Deep Learning | bilstm_fold_0      |          0 |              0 | See TensorBoard |

## 4. Technical Details
- **Dataset**: E. coli genomes (TaxID 562).
- **Task**: Multi-class classification of top 1000 gene products.
- **Loss Function**: CrossEntropyLoss.
- **Optimizer**: AdamW.

## 5. Observations
- **Transformer**: Captures long-range dependencies but requires more compute.
- **BiLSTM**: Efficient for sequences, good at local context.

## 6. Convergence
(Refer to TensorBoard logs for detailed loss curves)
