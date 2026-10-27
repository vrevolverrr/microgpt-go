# microgpt-go

**Note** This is a learning project. Minimal to no AI was used in the generation of the code. The goal is to learn neural networks and transformers from first principles. 

A scalar autograd engine in Go, built up from backprop to an MLP and a micro-transformer. Based on Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and microgpt.

```sh
make test
```

## Roadmap

- [x] **Autodiff and backprop:** a scalar `Value` that builds a DAG of operations and computes gradients in a backward pass.
- [ ] **MLP:** neurons, layers, and a multi-layer perceptron, with softmax to turn the output into probabilities.
- [ ] **Training:** a loss function, optimizers (SGD, Adam), and a loop that trains the MLP on a small dataset.
- [ ] **Tokenizer:** turn text into integer ids and back, starting at the character level.
- [ ] **Embeddings:** map each token id to a vector the model can learn.
- [ ] **Micro-transformer:** a tiny GPT that learns to generate text, built from attention, normalization, and an MLP block.
