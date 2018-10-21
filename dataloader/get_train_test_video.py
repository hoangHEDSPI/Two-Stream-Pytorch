import os, pickle

class UCF101_splitter():

    def __init__(self, path, split):
        self.path = path
        self.split = split

    def get_action_index(self):
        self.action_label = {}
        with open(self.path+"classInd.txt") as f:
            contents = f.readlines()
            contents = [x.strip('\r\n') for x in contents]
        for line in contents:
            index, label = line.split(" ")
            if index not in self.action_label.keys():
                self.action_label[index] =  label

    def file_2_dic(self, filename):
        with open(filename, "r") as f:
            content = f.readlines()
            content = [x.strip("\r\n") for x in content]
        dic = {}
        for line in content:
            video_name = line.split("/", 1)[1].split(" ", 1)[0]
            key = video_name.split("_",1)[1].split(".",1)[0]
            label = self.action_label[line.split("/")[0]]
            dic[key] = int(label)
        return dic




