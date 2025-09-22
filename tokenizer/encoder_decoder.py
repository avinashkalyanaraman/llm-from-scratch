import tokenizer
import sys
import settings
import regex as re
import pickle


def encode(text, vocab, merges):

    pretokens = tokenizer.split_based_on_specialtokens (settings.special_tokens, 
                                                        text, True)
    
    #this is now currently listed as <doc1>, <|endoftext|>, <doc2>, ...

    encoded = [] 

    #Precompute the vocab index of all special tokens so we can use that as we iterate through pretokens!
    special_token_vocab_idx = {} #Prefill this
    for special_token in settings.special_tokens:
        special_token_encoded = special_token.encode ("utf-8")
        
        #Finds the first (for us only) k for the corresponding matched v
        vocab_idx = next (k for k, v in vocab.items() if v == special_token_encoded) #generator
        #same as [k for k, v in vocab.items() if v == special_token_encoded][0]
        special_token_vocab_idx[special_token] = vocab_idx


    for text in pretokens:  

        #If special token:
        if text in settings.special_tokens:  
            print ("handling special token!") 
            encoded.append (special_token_vocab_idx[text])
            continue

        #not spl. token
        for match in re.finditer(settings.gpt2_pat, text): #match/pretoken will be (i)The, (ii) sky, (iii) is
            pretoken = match.group()
            print (f"handling pretoken {pretoken}")

            tokens_dict = {} #Just so that we can leverage prior merges() algorithm!

            numerical_pretoken = tuple(pretoken.encode("utf-8"))
            tokens_dict[numerical_pretoken] = 1

            for merge_pair, replacement in merges.items(): #insert order maintained post Python 3.7
                tokens_dict = tokenizer.merge(tokens_dict, merge_pair, replacement, False, None)

                # cant merge anymore. no point going through merges!
                key = next(iter(tokens_dict))
                if len(key) == 1:
                    break
                     
            
            #key = list(next(iter(tokens_dict)))
            encoded.extend (list(key))
                               
    return encoded

def decode (encoded_text, vocab):
    decoded_str = b''.join([vocab[ele] for ele in encoded_text])
    return decoded_str.decode("utf-8", errors="replace")


if __name__ == "__main__":
    text = sys.argv[1]

    #Read the stored vocab & merges:
    with open ("vocab_10000.pkl", "rb") as f:
        vocab = pickle.load(f)

    with open ("merges_10000.pkl", "rb") as f:
        merges = pickle.load(f)

    encoded_repsn = encode (text, vocab, merges)
    print (f"Encoded text = {encoded_repsn}")

    decoded_repsn = decode(encoded_repsn, vocab)
    print (f"Text after decoding = {decoded_repsn}")

    assert text == decoded_repsn

