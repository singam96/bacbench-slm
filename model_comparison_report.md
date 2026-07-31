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

3. **ProteinCNN**: A 1D Convolutional Neural Network with multiple kernel sizes.
   - Architecture: 4 parallel conv filters (kernels 3, 5, 7, 9), d_model=256
   - Mechanism: Multi-scale conv → global max pooling → concatenation → Linear Classifier.

4. **Random Forest**: A feature-based tree ensemble baseline.
   - Features: 1-mer, 2-mer, and 3-mer frequency vectors + log sequence length.
   - Algorithm: 300 trees with class-weight balancing (scikit-learn).

## 3. Results Summary

| Model         | Type                  | Fold               | Val Loss   |   Val Accuracy | Convergence              |
|:--------------|:----------------------|:-------------------|:-----------|---------------:|:-------------------------|
| transformer   | Deep Learning         | transformer_fold_0 | 0.3676     |         0.8544 | See charts below         |
| bilstm        | Deep Learning         | bilstm_fold_0      | 0.3022     |         0.8777 | See charts below         |
| cnn           | Deep Learning         | cnn_fold_0         | 0.1386     |         0.9237 | See charts below         |
| random_forest | Feature-based (k-mer) | all                | N/A        |         0.9693 | N/A (no training epochs) |
  - **Random Forest Val F1 (weighted)**: 0.9688

## 4. Technical Details
- **Dataset**: E. coli genomes (TaxID 562).
- **Task**: Multi-class classification of top 15 gene products.
- **Loss Function**: CrossEntropyLoss.
- **Optimizer**: AdamW.

## 5. Observations
- **Transformer**: Captures long-range dependencies but requires more compute.
- **BiLSTM**: Efficient for sequences, good at local context.
- **CNN**: Fast training, captures local sequence motifs (k-mer-like patterns) via multiple kernel sizes.
- **Random Forest**: Interpretable baseline using hand-crafted k-mer features; useful lower-bound reference.

## 6. Convergence
(Refer to TensorBoard logs for detailed loss curves)
