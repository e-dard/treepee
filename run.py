
from dtree import DecisionTree
from id3 import information_gain

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
keys = ["outlook", "temperature", "humidity", "wind", "target"]
data = [dict(zip(keys, x.split())) for x in values]

if __name__ == '__main__':
    tree = DecisionTree(data, keys[:-1], information_gain, target_name=keys[-1])
    from pprint import PrettyPrinter
    pp = PrettyPrinter()
    pp.pprint(tree.tree)

    print [(x[0]['target'], x[1]) for x in tree.classify(data)]
    