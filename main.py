import math
from random import random


class Net:
    net = []
    netC = []
    weight = []
    deltas = []
    struct = []

    def initNet(self, struct):
        self.struct = struct
        for i in range(len(struct)):
            self.weight.append([])
            self.deltas.append([])
            self.net.append([])
            self.netC.append([])
            self.net[i] = self.makeArray(struct[i], 0)
            self.netC[i] = self.makeArray(struct[i], 0)
            if(i == 0):
                continue
            self.deltas[i] = self.makeArray(struct[i], 0)
            ary = self.makeMatrix(struct[i], struct[i-1], 0)
            for a in range(len(ary)):
                for b in range(len(ary[a])):
                    ary[a][b] = self.random(-0.5, 0.5)
            self.weight[i] = ary

    def importWeight(self, w):
        self.weight = w

    def train(self, data, times, rate, moment):
        for t in range(times):
            errors = 0
            for i in range(len(data)):
                inputs = data[i][0]
                targets = data[i][1]
                o = self.update(inputs)
                error = self.backPropagate(targets, rate, moment)
                errors += error
                # print(error)
            if(t % 100 == 0):
                print(f"[ {t} ]", errors)

    def update(self, inputs):
        for i in range(len(self.struct)):
            if(i < 1):
                for j in range(self.struct[0]):
                    self.net[i][j] = inputs[j]
                continue
            for j in range(self.struct[i]):
                sum = 0
                for k in range(self.struct[i-1]):
                    sum += self.weight[i][j][k] * self.net[i-1][k]
                self.net[i][j] = self.sigmoid(sum)
        return self.net[-1]

    def backPropagate(self, targets, rate, moment):
        # 計算誤差
        for i in range(len(self.struct)-1, -1, -1):
            if(i < 1):
                continue
            for j in range(self.struct[i]):
                if(i == len(self.struct)-1):
                    self.deltas[i][j] = self.dsigmoid(
                        self.net[i][j]) * (targets[j] - self.net[i][j])
                    continue
                deltas = 0
                for k in range(self.struct[i+1]):
                    deltas += self.weight[i+1][k][j] * self.deltas[i+1][k]
                self.deltas[i][j] = self.dsigmoid(self.net[i][k]) * deltas
                #print("error: ", self.deltas[i][j])

        # 更新權重
        for i in range(len(self.struct)-1, -1, -1):
            if(i < 1):
                continue
            for j in range(self.struct[i]):
                change = self.deltas[i][j] * self.net[i][j]
                for k in range(self.struct[i-1]):
                    self.weight[i][j][k] -= rate * \
                        change + moment * self.netC[i][j]
                self.netC[i][j] = change

        # 計算總誤差
        error = 0
        for j in range(len(self.deltas[-1])):
            error += self.deltas[-1][j]
        return error

    def showWeight(self):
        print(self.weight)

    def tell(self, data):
        outpput = self.update(data)
        print("output: ", data, outpput)
        return outpput

    def makeArray(self, a, fill):
        ary = []
        for i in range(a):
            ary.append(fill)
        return ary

    def makeMatrix(self, a, b, fill):
        ary = []
        for i in range(a):
            ary.append(self.makeArray(b, fill))
        return ary

    def random(self, a, b):
        m = max(a, b)
        s = min(a, b)
        return (m - s) * random() + s

    def sigmoid(self, num):
        try:
            ans = math.exp(-num)
        except OverflowError:
            ans = float('inf')
        return (1/(1 + (ans)))

    def dsigmoid(self, num):
        return num * (1 - num)


# 網路架構
struct = [2, 2, 1]

# 資料
data = [
    [[1, 0], [1]],
    [[0, 1], [1]],
    [[1, 1], [0]],
    [[0, 0], [0]]
]


net = Net()
net.initNet(struct)
net.train(data, 100000, 0.01, 0.01)
net.showWeight()
net.tell([1, 0])
net.tell([0, 1])
net.tell([1, 1])
net.tell([0, 0])
