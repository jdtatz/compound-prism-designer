
def loadlibfunc(lib, name, ret, *args):
    func = getattr(lib, name)
    func.argtypes = args
    func.restype = ret
    return func
