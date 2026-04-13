import joblib as jb

def save_model(model, path):
    jb.dump(model, path)

def load_model(path):
    return jb.load(path)
