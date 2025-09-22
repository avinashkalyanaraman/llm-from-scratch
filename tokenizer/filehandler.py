import pickle

def serializedWrite(data, fname):
    # Write to a file
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def textWrite(data, fname):
    # Write to a file
    with open(fname, 'w') as f:
        for k, v in data.items():
            decoded_v = v.decode("utf-8", errors="replace")  # decode bytes to string
            f.write(f"{k},{decoded_v}\n")