### Start Up

First look into the `dataset_explore.ipynb` and get familiar with the data.

### Codes need your implementation

1. `op.py` 
   Implement the forward and backward function of `class Linear`
   Implement the `MultiCrossEntropyLoss`. Note that the `Softmax` layer could be included in the `MultiCrossEntropyLoss`.
   Try to implement `conv2D`, do not worry about the efficiency.
   You're welcome to implement other complicated layer (e.g.  ResNet Block or Bottleneck)
2. `models.py` You may freely edit or write your own model structure.
3. `mynn/lr_scheduler.py` You may implement different learning rate scheduler in it.
4. `MomentGD` in `optimizer.py`
5. Modifications in `runner.py` if needed when your model structure is slightly different from the given example.


### Train the model.

Run commands from the `codes/` directory.

MLP baseline:

```bash
python test_train.py --model mlp --optimizer sgd --scheduler multistep --epochs 5 --batch-size 64 --lr 0.06
```

CNN model:

```bash
python test_train.py --model cnn --optimizer sgd --scheduler multistep --epochs 5 --batch-size 64 --lr 0.02
```

Optimization direction with momentum:

```bash
python test_train.py --model mlp --optimizer momentum --scheduler multistep --epochs 5 --batch-size 64 --lr 0.03
```

The script saves checkpoints under `best_models/` and learning curves under `figs/`.

### Test the model.

Specify the model type and saved checkpoint:

```bash
python test_model.py --model mlp --checkpoint ./best_models/mlp_sgd_multistep/best_model.pickle
python test_model.py --model cnn --checkpoint ./best_models/cnn_sgd_multistep/best_model.pickle
```

### Analyze errors and visualize weights.

```bash
python weight_visualization.py --model mlp --checkpoint ./best_models/mlp_sgd_multistep/best_model.pickle
python weight_visualization.py --model cnn --checkpoint ./best_models/cnn_sgd_multistep/best_model.pickle
```

The script saves a confusion matrix, misclassified examples, and MLP weights or CNN kernels to `figs/`.


