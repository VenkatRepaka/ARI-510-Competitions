"""A small neural network (multi-layer perceptron, MLP) trained on the saved features, written in PyTorch.

WHAT THIS FILE DOES
    It defines MLPClassifier, a neural network that takes one row of features (e.g. EVA02 embedding + 74 quality
    numbers = 842 numbers) and outputs the probability that the image has an artifact.
    It has the same two methods as scikit-learn models, fit(X, y) and predict_proba(X), so baseline.py and
    make_submission.py can use it exactly like the SVM or logistic regression.
    Note: the network does NOT look at images. It learns from the features that quality_features.py and
    embed_features.py already computed; the big pretrained networks there stay frozen.

RESULTS
    CV F1 (baseline.py): 0.871-0.880,
    best "mlp_strong" on EVA02 + quality (CV 0.880, dev 0.873); slightly below the RBF SVM (0.886). With only
    about 750 training images per CV fold, the simpler SVM generalises at least as well.

THE NETWORK (one of the five, see "ENSEMBLE" below)
    input row (D numbers, standardised: each feature rescaled to mean 0, std 1)
      -> input dropout     randomly set 20% of the input numbers to 0 during training [1]
      -> Linear(D -> 256)  256 weighted sums of the inputs ("hidden units")
      -> GELU              a smooth non-linearity [2]; without it the network would just be one weighted sum
      -> dropout 50%       randomly switch off half of the hidden units during training [1]
      -> Linear(256 -> 1)  one number, the "logit" (log-odds of artifact)
      -> sigmoid           turns the logit into a probability between 0 and 1 (only when predicting)
    Dropout makes the network not rely on any single feature or unit, which reduces overfitting on small
    datasets. It is switched off when predicting.

ENSEMBLE OF 5 NETWORKS
    Five networks with different random starting weights are trained, and their probabilities averaged.
    Averaging several independently trained networks gives more stable and usually better predictions than
    one network [7][8].
    Speed trick: instead of a Python loop over five nn.Linear networks, the five are stored as stacked weight
    tensors (shape (5, ...)) and run together with batched matrix multiplication (torch.bmm). The loss is the
    SUM of the five networks' losses; since each network's loss only depends on its own weights, each network
    still gets exactly the gradient it would get alone. Running an ensemble's models together as stacked
    weights is the "model ensembling" pattern described in the PyTorch tutorials [10]. What the five share: the same mini-batches in the same
    order. What differs: starting weights and dropout masks. This took one fit from about 60 s to about 14 s.

TRAINING
    loss       binary cross-entropy on the logit (BCEWithLogitsLoss: the logistic-regression loss, computed in a
               numerically stable way)
    optimiser  AdamW [4][5]: Adam with weight decay (wd) applied directly to the weights, pulling them towards 0
               (another protection against overfitting)
    schedule   learning rate starts at lr = 0.001 and decreases along a cosine curve to 0 over the epochs [6]
    epochs     a FIXED 60 passes over the training data, mini-batches of 64 images.
               No early stopping: early stopping would need held-out data to decide when to stop, and in CV that
               data must stay unseen.
    The three variants used in baseline.py:
        mlp         hidden 256, input dropout 0.2, weight decay 0.01   (the defaults below)
        mlp_h64     hidden 64                                          (a smaller network)
        mlp_strong  input dropout 0.5, weight decay 0.1                (stronger regularisation; the best)

REPRODUCIBILITY
    Runs on the CPU with fixed seeds (starting weights, dropout and batch order), so the same data always gives
    exactly the same predictions.

REFERENCES
    [1] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. (2014). Dropout: A simple
        way to prevent neural networks from overfitting. Journal of Machine Learning Research 15, 1929-1958.
        (Dropout on hidden units and on the inputs.)
    [2] Hendrycks, D., Gimpel, K. (2016). Gaussian error linear units (GELUs). arXiv:1606.08415.
    [3] Glorot, X., Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
        Proc. AISTATS 2010. (Scaling the starting weights by the number of inputs.)
    [4] Kingma, D. P., Ba, J. (2015). Adam: A method for stochastic optimization. Proc. ICLR 2015.
    [5] Loshchilov, I., Hutter, F. (2019). Decoupled weight decay regularization (AdamW). Proc. ICLR 2019.
    [6] Loshchilov, I., Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. Proc. ICLR 2017.
        (The cosine learning-rate schedule.)
    [7] Dietterich, T. G. (2000). Ensemble methods in machine learning. Proc. Multiple Classifier Systems,
        LNCS 1857, 1-15.
    [8] Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). Simple and scalable predictive uncertainty
        estimation using deep ensembles. Proc. NeurIPS 2017. (Averaging networks trained from different
        random starts.)
    [9] Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance deep learning library.
        Proc. NeurIPS 2019.
    [10] PyTorch tutorial: Model ensembling (torch.func: stack the models' weights and run them together).
        https://pytorch.org/tutorials/intermediate/ensembling.html
"""
import numpy as np
import torch
from torch import nn


# ----------------------------------------------------------------------------------------------------------
# The five networks, stored and run together
# ----------------------------------------------------------------------------------------------------------
class _Ensemble(nn.Module):
    """n networks of the form  dropout -> Linear(d, h) -> GELU -> dropout -> Linear(h, 1),  in one module.

    The leading "_" in the name means it is only used inside this file (by MLPClassifier).
    Arguments: n = number of networks (5), d = number of input features, h = hidden units (256),
               in_drop / drop = dropout rates on the inputs / on the hidden units.
    """

    def __init__(self, n, d, h, in_drop, drop):
        super().__init__()   # required set-up of every PyTorch module
        # nn.Parameter marks a tensor as "learnable": the optimiser will update it.
        # The first dimension (n) indexes the network: w1[0] belongs to network 0, w1[1] to network 1, ...
        self.w1 = nn.Parameter(torch.empty(n, d, h))   # layer 1 weights: d inputs -> h hidden units, per network
        self.b1 = nn.Parameter(torch.zeros(n, 1, h))   # layer 1 biases (start at 0)
        self.w2 = nn.Parameter(torch.empty(n, h, 1))   # layer 2 weights: h hidden units -> 1 output, per network
        self.b2 = nn.Parameter(torch.zeros(n, 1, 1))   # layer 2 bias (start at 0)
        # Starting weights: random, uniform between -1/sqrt(fan_in) and +1/sqrt(fan_in), where fan_in = number
        # of inputs to the layer. This is the same scale nn.Linear uses by default, and keeps the size of the
        # signals roughly constant through the layers at the start of training [3]. Each network gets its own
        # random numbers, which is what makes the five networks different.
        for w, fan_in in ((self.w1, d), (self.w2, h)):  # same scale as nn.Linear's default init
            nn.init.uniform_(w, -fan_in ** -0.5, fan_in ** -0.5)
        self.in_drop, self.drop = nn.Dropout(in_drop), nn.Dropout(drop)

    def forward(self, x):
        """x: a batch of B input rows, shape (B, D)  ->  logits, shape (n, B): one per network per row."""
        # unsqueeze(0): (B, D) -> (1, B, D); expand(n, -1, -1): view it as (n, B, D), the same rows for every
        # network (no copying). Dropout then draws a separate random mask for each network.
        x = self.in_drop(x.unsqueeze(0).expand(self.w1.shape[0], -1, -1))
        # torch.bmm = batched matrix multiplication: network k computes x[k] @ w1[k], all n at once.
        # (n, B, D) @ (n, D, h) -> (n, B, h); + bias; GELU non-linearity; dropout on the hidden units.
        h = self.drop(nn.functional.gelu(torch.bmm(x, self.w1) + self.b1))
        # (n, B, h) @ (n, h, 1) -> (n, B, 1); + bias; squeeze(-1) drops the last size-1 dimension -> (n, B).
        return (torch.bmm(h, self.w2) + self.b2).squeeze(-1)


# ----------------------------------------------------------------------------------------------------------
# The classifier used by baseline.py and make_submission.py
# ----------------------------------------------------------------------------------------------------------
class MLPClassifier:
    """Scikit-learn style wrapper: MLPClassifier(...).fit(X, y).predict_proba(X_new)."""

    def __init__(self, hidden=256, in_drop=0.2, drop=0.5, lr=1e-3, wd=1e-2, epochs=60, batch=64, n_seeds=5, seed=0):
        # Settings (see the header):
        #   hidden   hidden units per network          in_drop  dropout rate on the inputs
        #   drop     dropout rate on hidden units      lr       starting learning rate
        #   wd       weight decay (AdamW)              epochs   passes over the training data
        #   batch    images per mini-batch             n_seeds  number of networks averaged
        #   seed     random seed
        # locals() is a dictionary of all the arguments above (plus "self"); copying it into self.__dict__
        # stores each one as an attribute (self.hidden, self.lr, ...). "self" itself is then removed.
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def fit(self, X, y):
        """Train on X (numpy array, images x features) and y (0/1 labels). Returns self."""
        # Fix the randomness: torch.manual_seed for starting weights and dropout; a separate generator g for the
        # order of the mini-batches.
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        # Standardise: each feature -> mean 0, std 1, using the TRAINING data only (the same mu / sd are reused
        # in predict_proba, so held-out data never influences them). + 1e-6 avoids dividing by zero for a
        # feature that is constant.
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        # Convert to PyTorch tensors (32-bit numbers).
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32)
        self.net = _Ensemble(self.n_seeds, X.shape[1], self.hidden, self.in_drop, self.drop)
        # AdamW updates every learnable parameter; weight_decay pulls weights towards 0 at each step [5].
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        # Cosine schedule: the learning rate goes from lr down to 0 over `epochs` steps (one step per epoch) [6].
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        # Binary cross-entropy computed from logits (stable version of "sigmoid, then log-loss").
        # reduction="sum": add up the losses of all networks and all rows (divided by the batch size below).
        loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        self.net.train()   # training mode: dropout ON
        for _ in range(self.epochs):
            # One epoch: shuffle the row numbers (randperm) and cut them into mini-batches of `batch` rows.
            for idx in torch.randperm(len(Xt), generator=g).split(self.batch):
                logits = self.net(Xt[idx])   # (5, batch size): each network's prediction for each row
                opt.zero_grad()              # clear the gradients from the previous step
                # yt[idx].expand_as(logits): the same labels for every network.
                # Sum over networks and rows, divided by the number of rows = the SUM over networks of each
                # network's average loss, so each network is trained as if it were alone.
                # backward() computes the gradients (backpropagation).
                (loss_fn(logits, yt[idx].expand_as(logits)) / len(idx)).backward()
                opt.step()                   # update the weights
            sched.step()                     # lower the learning rate for the next epoch
        self.net.eval()    # prediction mode: dropout OFF
        return self

    @torch.no_grad()   # no training here, so PyTorch does not need to track gradients (saves time and memory)
    def predict_proba(self, X):
        """Probabilities for new rows X. Returns shape (rows, 2): [P(clean), P(artifact)], like scikit-learn."""
        # Standardise with the TRAINING mean / std saved in fit().
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        # sigmoid: logits -> probabilities, shape (5, rows); .mean(0) averages the 5 networks -> (rows,)
        p = torch.sigmoid(self.net(Xt)).mean(0).numpy()
        # Two columns, as scikit-learn returns them; baseline.py uses column 1 = P(artifact).
        return np.column_stack([1 - p, p])
