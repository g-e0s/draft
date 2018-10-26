import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import numpy as np


class NetworkTrainer:
    def __init__(self, train_dataset, validation_dataset=None, batch_size=128, log_interval=100):
        self.cuda = torch.cuda.is_available()
        self.log_interval = log_interval
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        self.train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        self.validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, **kwargs) \
            if validation_dataset else None

    def set_optimizer(self, model, lr=1e-3):
        return Adam(model.parameters(), lr=lr)

    def set_scheduler(self, optimizer, step_size=8, gamma=0.1, last_epoch=-1):
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)

    def fit_model(self, model, loss, lr, start_epoch=0, n_epochs=20, metrics=[]):
        if self.cuda:
            model.cuda()
        optimizer = self.set_optimizer(model, lr)
        scheduler = self.set_scheduler(optimizer)

        for epoch in range(0, start_epoch):
            scheduler.step()

        for epoch in range(start_epoch, n_epochs):
            scheduler.step()

            # Train stage
            train_loss, metrics = self.train_epoch(model, loss, optimizer, metrics)

            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            val_loss, metrics = self.test_epoch(model, loss, metrics)
            val_loss /= len(self.validation_data_loader)

            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                     val_loss)
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)

        return model

    def train_epoch(self, model, loss_fn, optimizer, metrics):
        for metric in metrics:
            metric.reset()

        model.train()
        losses = []
        total_loss = 0

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if self.cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            optimizer.zero_grad()
            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

            if batch_idx % self.log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data[0]), len(self.train_data_loader.dataset),
                    100. * batch_idx / len(self.train_data_loader), np.mean(losses))
                for metric in metrics:
                    message += '\t{}: {}'.format(metric.name(), metric.value())

                print(message)
                losses = []

        # average loss
        total_loss /= (batch_idx + 1)
        return total_loss, metrics

    def test_epoch(self, model, loss, metrics, ):
        if self.validation_data_loader:
            with torch.no_grad():
                for metric in metrics:
                    metric.reset()
                model.eval()
                val_loss = 0
                for batch_idx, (data, target) in enumerate(self.validation_data_loader):
                    target = target if len(target) > 0 else None
                    if not type(data) in (tuple, list):
                        data = (data,)
                    if self.cuda:
                        data = tuple(d.cuda() for d in data)
                        if target is not None:
                            target = target.cuda()

                    outputs = model(*data)

                    if type(outputs) not in (tuple, list):
                        outputs = (outputs,)
                    loss_inputs = outputs
                    if target is not None:
                        target = (target,)
                        loss_inputs += target

                    loss_outputs = loss(*loss_inputs)
                    batch_loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                    val_loss += batch_loss.item()

                    for metric in metrics:
                        metric(outputs, target, loss_outputs)

            return val_loss, metrics