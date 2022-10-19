import pandas as pd


class FileSave:
    def __init__(self, market):
        self.market = market

    def save(self, array, name):
        with open("bi-data/%s/%s" % (self.market, name), "w+") as file:
            for element in array:
                if not pd.isnull(element):
                    file.write("%s\n" % float(element))
