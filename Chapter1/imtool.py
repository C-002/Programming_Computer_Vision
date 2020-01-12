import os
def get_imlist(path):
    """ return all JPG image filenames in the path """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
