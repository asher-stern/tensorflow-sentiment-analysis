class Averager:

    def __init__(self, size):
        self._l = []
        self._size = size

    def add(self, v):
        if len(self._l) == self._size:
            self._l.pop(0)
        self._l.append(float(v))

    def average(self):
        return sum(self._l) / len(self._l)
