'''
This code reads a parent jsonl file, and
    (i) splits it into train and valdn
    (ii) Writes train to _train.jsonl and validn to _valdn.jsonl
'''

from sklearn.model_selection import train_test_split
import utils


if __name__ == '__main__':

    SEED = 42
    data_file = 'data/sft.jsonl'
    train_output_file = 'data/sft_train.jsonl'
    valn_output_file = 'data/sft_valdn.jsonl'

    dataset = utils.readJSONL (data_file)
    print (f"Total Dataset size = {len(dataset)}")

    #Split the data!
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=SEED)    
    #train_data = train_data[0:1000]
    #val_data = val_data[0:8]

    print (f"Train data len = {len(train_data)}")
    print (f"Val data len = {len(val_data)}")

    utils.writeJSONL(train_output_file, train_data)
    utils.writeJSONL(valn_output_file, val_data)

    print(f"Wrote {train_output_file}")
    print(f"Wrote {valn_output_file}")


    