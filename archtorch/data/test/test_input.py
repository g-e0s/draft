import pytest
import unittest.mock
from data.input import FileParser, SiameseData
import glob

path = '/ext/data/'
extension = '.txt'
files_list = ['1_1', '1_2', '2_1', '2_2']
fp = FileParser(extension=extension, sep=' ')

sd = SiameseData(fp, path)
sd.data_idx = files_list


def test_parse_string():
    fs = ' [ 0.01 -0.21 45.1 ] '
    expected = [0.01, -0.21, 45.1]
    parsed = fp.parse_string(fs)
    assert expected == parsed


def test_parse_filename():
    file = 'vjdfbvjs/vdbhjbvjd/vfjdbvjhd/vdjfb_sjbfd.csgvdc'
    assert fp.parse_filename(file) == 'vdjfb_sjbfd'


def test_get_filename_for_idx():
    idx = '1_a'
    assert '/ext/data/1_a.txt' == fp.get_filename_for_idx(path, idx)


def test_get_files_list(monkeypatch):
    expected = [path + x + extension for x in files_list]
    monkeypatch.setattr('glob.glob', lambda path: [path[:-1] + x + extension for x in files_list])
    assert fp.get_files_list(path) == expected


def test_register_data(monkeypatch):
    expected = ['1_1', '1_2', '2_1', '2_2']
    monkeypatch.setattr('glob.glob', lambda path: [path[:-1] + x + extension for x in files_list])
    assert fp.register_data(path) == expected


def test_get_positive_indices():
    assert sd.get_positive_indices('1_1') == ['1_1', '1_2']


def test_get_negative_indices():
    assert sd.get_negative_indices('1_1') == ['2_1', '2_2']


def test_sample_indices():
    assert sd.sample_indices(0) == (('1_1', '1_2'), 0) or (('1_1', '2_1'), 1) or (('1_1', '2_2'), 1)
