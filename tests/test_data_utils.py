import pytest
import os, sys
sys.path.append(os.getcwd())
sys.path.append('..')
import data_utils
import main

def test_create_dataset_reallabor_passes_with_default_args():
    args = main.get_default_args()
    data_path = os.path.join(data_utils.dataset_path(1, 'processed_csv_no_con'), '11228_12.csv')
    args, train_dataset, test_dataset = data_utils.create_dataset_reallabor(args, data_path)

def test_easy_dataset_reallabor_passes():
    data_utils.easy_reallabor_dataset(1, 12, 'processed_csv_no_con', -20)


if __name__=='__main__':
    pass