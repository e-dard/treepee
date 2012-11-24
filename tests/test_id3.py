import unittest
import id3

class TestID3(unittest.TestCase):

    def setUp(self):
        values = ["sunny hot high weak no",
                  "sunny hot high strong no",
                  "overcast hot high weak yes",
                  "rain mild high weak yes",
                  "rain cool normal weak yes",
                  "rain cool normal strong no",
                  "overcast cool normal strong yes",
                  "sunny mild high weak no",
                  "sunny cool normal weak yes",
                  "rain mild normal weak yes",
                  "sunny mild normal strong yes",
                  "overcast mild high strong yes",
                  "overcast hot normal weak yes",
                  "rain mild high strong no"]
        keys = ["outlook", "temperature", "humidity", "wind", "output"]
        self.attrs = keys[:-1]
        self.target = keys[-1]
        self.data = [dict(zip(keys, x.split())) for x in values]

    def test__entropy(self):
        """ Tests id3._entropy """
        data = [{'foo': x} for x in "sample_text"]
        self.assertAlmostEquals(id3._entropy(data, 'foo'), 3.0958, places=4)

    def test_information_gain(self):
        """ Tests id3.information_gain """

        gain = id3.information_gain
        actual = [gain(self.data, atr, self.target) for atr in self.attrs]
        expected = [0.246, 0.029, 0.151, 0.048]
        for exp, act in zip(expected, actual):
            self.assertAlmostEquals(exp, act, places=2)


    