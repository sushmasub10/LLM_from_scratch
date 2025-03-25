1️. Installed Dependencies
Checked if Lightning is installed and installed it if missing.
Imported PyTorch, Lightning, and essential modules like torch.nn, torch.optim, DataLoader, etc.

2. Implemented Positional Encoding
Created a PositionEncoding class to add sinusoidal positional encoding to embeddings.
Used sin for even indices and cos for odd indices.
Used register_buffer() to store pe without affecting gradients.

3. Implemented Scaled Dot-Product Attention
Built an Attention class with learnable weight matrices W_q, W_k, W_v.
Computed similarity scores:
\text{scaled_sims} = \frac{q \cdot k^T}{\sqrt{d_{model}}}
Applied masked softmax for decoding.
Used the mask to prevent information leakage (future tokens are not seen).

4. Built a Decoder-Only Transformer Model
Defined a DecoderOnlyTransformer class using LightningModule.
Components:
Embedding layer (nn.Embedding)
Positional encoding (PositionEncoding)
Masked self-attention (Attention)
Feedforward (nn.Linear)
Used residual connections after self-attention.
Configured Adam optimizer for training.

5. Defined Tokenization and Vocabulary
Created token-to-ID mapping (token_to_id) and reverse mapping (id_to_token).
Tokens: what, is, apple, healthy, <EOS>
Prepared input sequences as tokenized tensors.

6. Created Target Labels for Training
Designed target labels by shifting inputs:
Example:
Input: what is apple <EOS>
Target: is apple <EOS> healthy
This follows the Teacher Forcing method:
During training, the model is given correct previous tokens instead of its own predictions.
Helps faster convergence by guiding the model toward correct outputs.

7. Created a DataLoader
Wrapped input-label pairs into a TensorDataset.
Used DataLoader to batch data and feed it into the model.

8. Trained the Transformer Model
Used Lightning's Trainer for training (max_epochs=30).
Implemented a training_step function:
Computed predictions.
Calculated CrossEntropyLoss for classification.
Used gradient descent (Adam optimizer, lr=0.1) to update weights.

9.Generated Predictions Using Autoregressive Decoding
Autoregressive decoding:
Predicted tokens one by one (instead of generating all at once).
Used previously predicted tokens as input for the next step.
Stopped prediction at <EOS> token.

10.  Converted Predictions Back to Words
Converted predicted token IDs back to words using id_to_token.
Printed the final output sequence.


Key Takeaways from Your Implementation
1. Transformer Type:
Decoder-Only Transformer (like GPT)
Uses masked self-attention to prevent peeking at future tokens.

2. Important Parameters:
d_model=2 → Small embedding dimension (usually higher in real cases, like 512 or 1024).
max_len=6 → Max sequence length.
num_tokens=4 → Vocabulary size (excluding <EOS>).
lr=0.1 → Learning rate for Adam optimizer.

3. Teacher Forcing:
Used actual previous token during training instead of model's own prediction.
Prevents compounding errors early in training.

4. Autoregressive Decoding:
Greedy decoding (argmax() used at each step).
Stops at <EOS>.


 Next Steps for Improvement
1️ Increase d_model for better feature representation.
2️ Use a Multi-Head Attention layer for richer context.
3️ Experiment with different training data for more complex sentence generation.
4️ Implement beam search instead of greedy decoding for better predictions.




