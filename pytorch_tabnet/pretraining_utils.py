from torch.utils.data import DataLoader
from pytorch_tabnet.utils import (create_sampler,
                                  PredictDataset,
                                  TorchDataset)


def create_dataloaders(
    X_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory
):
    """
    Create dataloaders with or wihtout subsampling depending on weights and balanced.

    Parameters
    ----------
    X_train : np.ndarray
        Training data
    eval_set : list of tuple
        List of eval tuple set (X, y)
    weights : either 0, 1, dict or iterable
        if 0 (default) : no weights will be applied
        if 1 : classification only, will balanced class with inverse frequency
        if dict : keys are corresponding class values are sample weights
        if iterable : list or np array must be of length equal to nb elements
                      in the training set
    batch_size : int
        how many samples per batch to load
    num_workers : int
        how many subprocesses to use for data loading. 0 means that the data
        will be loaded in the main process
    drop_last : bool
        set to True to drop the last incomplete batch, if the dataset size is not
        divisible by the batch size. If False and the size of dataset is not
        divisible by the batch size, then the last batch will be smaller
    pin_memory : bool
        Whether to pin GPU memory during training

    Returns
    -------
    train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
        Training and validation dataloaders
    """
    need_shuffle, sampler = create_sampler(weights, X_train)

    train_dataloader = DataLoader(
        PredictDataset(X_train),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=need_shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory
    )

    valid_dataloaders = []
    for X, Y in eval_set:
        valid_dataloaders.append(
            DataLoader(
                TorchDataset(X, X),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        )

    return train_dataloader, valid_dataloaders
