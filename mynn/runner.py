import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        eval_interval = kwargs.get("eval_interval", log_iters)
        eval_batch_size = kwargs.get("eval_batch_size", self.batch_size)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            self._set_training(True)
            num_batches = int(np.ceil(X.shape[0] / self.batch_size))

            for iteration in range(num_batches):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]
                if train_X.shape[0] == 0:
                    continue

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                if (iteration + 1) % eval_interval == 0 or iteration == num_batches - 1:
                    dev_score, dev_loss = self.evaluate(dev_set, batch_size=eval_batch_size)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)

                    if dev_score > best_score:
                        save_path = os.path.join(save_dir, 'best_model.pickle')
                        self.save_model(save_path)
                        print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                        best_score = dev_score

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    if len(self.dev_scores) > 0:
                        print(f"[Dev] loss: {self.dev_loss[-1]}, score: {self.dev_scores[-1]}")
        self.best_score = best_score

    def evaluate(self, data_set, batch_size=None):
        X, y = data_set
        batch_size = batch_size or X.shape[0]
        self._set_training(False)
        total_loss = 0
        total_score = 0
        total_num = 0

        for start in range(0, X.shape[0], batch_size):
            batch_X = X[start:start + batch_size]
            batch_y = y[start:start + batch_size]
            logits = self.model(batch_X)
            loss = self.loss_fn(logits, batch_y)
            score = self.metric(logits, batch_y)
            total_loss += loss * batch_X.shape[0]
            total_score += score * batch_X.shape[0]
            total_num += batch_X.shape[0]

        self._set_training(True)
        return total_score / total_num, total_loss / total_num
    
    def save_model(self, save_path):
        self.model.save_model(save_path)

    def _set_training(self, training):
        for layer in getattr(self.model, 'layers', []):
            if hasattr(layer, 'training'):
                layer.training = training
