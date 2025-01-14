# logical circuit by perceptrons.

import numpy as np

class LogicalCircuit:
    def __init__(self):
        pass

    def AND(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.7
        tmp = np.sum(w*x) +b
        if tmp <= 0:
            return 0
        else:
            return 1

    def OR(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.3
        tmp = np.sum(x*w) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def NAND(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([-0.5,-0.5])
        b = 0.7
        tmp = np.sum(x*w) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def XOR(self, x1, x2):  # similar with composite function
        x3 = self.NAND(x1, x2)
        x4 = self.OR(x1, x2)
        res = self.AND(x3, x4)
        return res


if __name__ == "__main__":
    f = LogicalCircuit()
    print("x1  x2  y")
    print(f"AND\n"
          f"0\t0\t{f.AND(0, 0)}\n"
          f"1\t0\t{f.AND(1, 0)}\n"
          f"0\t1\t{f.AND(0, 1)}\n"
          f"1\t1\t{f.AND(1, 1)}\n")
    print(f"NAND\n"
          f"0\t0\t{f.NAND(0, 0)}\n"
          f"1\t0\t{f.NAND(1, 0)}\n"
          f"0\t1\t{f.NAND(0, 1)}\n"
          f"1\t1\t{f.NAND(1, 1)}\n")
    print(f"OR\n"
          f"0\t0\t{f.OR(0, 0)}\n"
          f"1\t0\t{f.OR(1, 0)}\n"
          f"0\t1\t{f.OR(0, 1)}\n"
          f"1\t1\t{f.OR(1, 1)}\n")
    print(f"XOR\n"
          f"0\t0\t{f.XOR(0, 0)}\n"
          f"1\t0\t{f.XOR(1, 0)}\n"
          f"0\t1\t{f.XOR(0, 1)}\n"
          f"1\t1\t{f.XOR(1, 1)}\n")
