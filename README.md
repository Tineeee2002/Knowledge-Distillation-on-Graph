# Knowledge Distillation on Graph Neural Networks

This project demonstrates a knowledge distillation approach for graph neural networks (GNNs), focusing on preserving local structures while training a student model to mimic a teacher model.

## Overview

Knowledge distillation transfers knowledge from a larger, pre-trained **teacher model** to a smaller **student model**. In this project, we:

1. Train a teacher GNN on the Cora dataset.
2. Train a student GNN using both:
   - Knowledge from the teacher (via distillation loss).
   - Local structure preservation (via structural similarity loss).
3. Evaluate and compare the performance of the teacher and student models.

## Key Components

### Models
- **Teacher GNN**: A deeper GCN with three layers.
- **Student GNN**: A shallower GCN with two layers.

### Loss Functions
1. **Task Loss**: Cross-entropy loss on the labeled training nodes.
2. **Knowledge Distillation Loss**: KL divergence between the softened logits of the teacher and student.
3. **Local Structure Preservation Loss**: MSE loss to preserve node similarity in the local subgraph structures.

### Dataset
- **Cora**: A citation network dataset where nodes represent papers and edges represent citations.

### Frameworks and Libraries
- PyTorch
- PyTorch Geometric

## File Details

The provided code:
- Loads the Cora dataset.
- Defines the teacher and student GCN architectures.
- Implements the local structure-preserving knowledge distillation method.
- Trains and evaluates both teacher and student models.

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric

Install the required dependencies using:
```bash
pip install -r requirements.txt
```

### Example Requirements File
```
torch
torch-geometric
numpy
scipy
```

## Results

Expected outputs from training and evaluation:

- **Teacher Model**:
  - High accuracy on training and test sets, demonstrating strong performance.
- **Student Model**:
  - Comparable accuracy with reduced complexity, showing effective knowledge transfer.

## Notes

- You can adjust hyperparameters such as temperature, hidden dimensions, and the number of hops to fine-tune performance.
- The dataset splitting follows the standard split used in GNN literature.

## References

1. [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
2. [Cora Dataset Description](https://relational.fit.cvut.cz/dataset/CORA)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

------------

Enjoy exploring Knowledge Distillation on Graph with this implementation!


