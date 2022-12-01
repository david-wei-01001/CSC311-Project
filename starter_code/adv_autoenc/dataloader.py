import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np


class InteractionDataset(Dataset):
    def __init__(self, data, meta=False, **kwargs):
        self.data: pd.DataFrame = data
        self.meta: bool = meta
        if meta:
            self.question_meta: pd.DataFrame = kwargs["question_meta"]
            self.student_meta: pd.DataFrame = kwargs["student_meta"]
            self.subject_meta: pd.DataFrame = kwargs["subject_meta"]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # return the row for user in the sparse matrix
        row = self.data.iloc[idx]
        row = torch.from_numpy(row.values).float()

        if self.meta:
            # get index for non NaN questions
            question_idx = np.where(~np.isnan(row))[0]

            # get the question metadata
            question_meta = self.question_meta[question_idx]
            # get the subject metadata
            subject_meta = self.subject_meta[question_meta[:, 1]]
            # get the student metadata
            student_meta = self.student_meta[idx]

            # convert pd dataframe to tensor
            question_meta = torch.from_numpy(question_meta).float()
            subject_meta = torch.from_numpy(subject_meta).float()
            student_meta = torch.from_numpy(student_meta).float()

            return row, question_meta, subject_meta, student_meta

        return row


class ObservationDataset(Dataset):
    def __init__(self, data, meta=False, **kwargs):
        self.data: pd.DataFrame = data
        self.meta: bool = meta
        if meta:
            self.question_meta: pd.DataFrame = kwargs["question_meta"]
            self.student_meta: pd.DataFrame = kwargs["student_meta"]
            self.subject_meta: pd.DataFrame = kwargs["subject_meta"]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # user_id, question_id, is_correct
        row = self.data.iloc[idx]

        if self.meta:

            # get the question metadata
            question_meta = self.question_meta[row["question_id"]]
            # get the subject metadata
            subject_meta = self.subject_meta[question_meta[1]]
            # get the student metadata
            student_meta = self.student_meta[row["user_id"]]

            # convert pd dataframe to tensor
            question_meta = torch.from_numpy(question_meta).float()
            subject_meta = torch.from_numpy(subject_meta).float()
            student_meta = torch.from_numpy(student_meta).float()

            row = torch.from_numpy(row.values).int()
            return row, question_meta, subject_meta, student_meta

        row = torch.from_numpy(row.values).int()
        return row


def load_dataset(meta=False):
    """ Load cvs files and return train, valid, test datasets.
    """
    train_data = np.load('../data/imputed_matrix.npz')["arr_0"]
    # transform to pandas dataframe
    train_data = pd.DataFrame(train_data)

    val_data = pd.read_csv("../data/valid_data.csv")
    test_data = pd.read_csv("../data/test_data.csv")

    # Load meta data
    question_meta = pd.read_csv("../data/question_meta.csv", index_col=0)
    student_meta = pd.read_csv("../data/student_meta.csv", index_col=0)
    subject_meta = pd.read_csv("../data/subject_meta.csv", index_col=0)

    train_dataset = InteractionDataset(train_data,
                                       meta=meta,
                                       question_meta=question_meta,
                                       student_meta=student_meta,
                                       subject_meta=subject_meta)
    val_dataset = ObservationDataset(val_data,
                                     meta=meta,
                                     question_meta=question_meta,
                                     student_meta=student_meta,
                                     subject_meta=subject_meta)
    test_dataset = ObservationDataset(test_data,
                                      meta=meta,
                                      question_meta=question_meta,
                                      student_meta=student_meta,
                                      subject_meta=subject_meta)

    return train_dataset, val_dataset, test_dataset
