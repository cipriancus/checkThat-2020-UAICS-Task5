from unittest import TestCase
from os.path import join, dirname

from .task5 import check_format

_ROOT_DIR = dirname(dirname(__file__))
_TEST_DATA_FOLDER = join(_ROOT_DIR, 'format_checker/data')


class FormatCheckerTask5(TestCase):
    _OK_FILES = ['task5_OK.txt']
    _NOT_OK_FILES = ['task5_NOTOK_0.txt', 'task5_NOTOK_MISSING_ID.txt', 'task5_NOTOK_DUP_LINE_NUM.txt']

    def test_ok(self):
        for _file in self._OK_FILES:
            self.assertTrue(check_format(join(_TEST_DATA_FOLDER, _file)))

    def test_not_ok(self):
        for _file in self._NOT_OK_FILES:
            self.assertFalse(check_format(join(_TEST_DATA_FOLDER, _file)))
