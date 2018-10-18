import os, pickle

class UCF101_splitter():

    def __init__(self, path):
        self.path = path

    def get_action_index(self):
        self.action_label = {}
        with open(self.path+"classInd.txt") as f:
            contents = f.readlines()
            contents = [x.strip('\r\n') for x in contents]
        for line in contents:
            index, label = line.split(" ")
            if index not in self.action_label.keys():
                self.action_label[index] =  label


