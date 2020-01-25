import numpy as np

def tsg_add(a, b):
    return a + b

    
def _tsg_add_underscore(a, b):
    return a + b

def __tsg_add_double_underscore(a, b):
    return a + b

class A:
    def get_1(self):
        return 1
 
    def _get_2(self):
        return 2
 
    def __get_3(self):
        return 3

if __name__=='__main__':
    print(tsg_add(3, 5))
    print(dir(A))
    a = A()
    print(a._get_2())
    print(a._A__get_3())
