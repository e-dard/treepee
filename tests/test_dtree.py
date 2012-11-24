import unittest

import dtree

class TestDtree(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_frequencies(self):
        """ Tests dtree.frequencies """
        input_ = [1, 2, 2, 3]
        expected = [(1, 1), (2, 2), (3, 1)]
        actual = dtree.frequencies(input_)
        self.assertEquals(set(expected), set(actual))

    def test_majority_vote(self):
        """ Tests dtree.majority_vote """
        input_ = [1, 2, 2, 3]
        input_ = [{'foo': x} for x in input_]
        self.assertEquals(2, dtree.majority_vote(input_, attribute='foo'))

    def test__choose_attributes(self):
        """ Tests dtree._choose_attributes """
        fitness = lambda x, y, _: x[y]
        data = {'a': 1, 'b': 2, 'c': 0}
        act = dtree._choose_attribute(data, ['a', 'b', 'c'], None, fitness)
        self.assertEquals('b', act)

    