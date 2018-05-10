import numpy as np

class hrKNN:
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def forceCalc(self, x, k):
        # calculate the number of classes
        nums = set()
        for ele in self.labels:
            nums.add(ele)
        N = len(nums)
        dist = dict()
        if not isinstance(self.datas, np.ndarray):
            self.datas = np.array(self.datas)

        for i in range(self.datas.shape[0]):
            dist[i] = self.__euclideanDist(self.datas[i], x)

        sorted(dist.items(), lambda item : item[1])    # sorted by distance

        cls = [0 for i in range(N)]
        key = dist.keys()
        for i in range(k):
            cls[key[i]] = cls[key[i]] + 1

        return cls.index(max(cls))

    # TODO: add more efficient ways using kd

    def __euclideanDist(self, x, y):
        t = 0.
        for i in x:
            for j in y:
                t = t + (x - y)**2
        return t



class hrKD:
    def __init__(self):
        pass
