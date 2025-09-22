import regex as re
import chunker
from multiprocessing import Process, Queue
import time
from collections import defaultdict
import filehandler   
import settings 


def init_vocab():
    return {i: bytes([i]) for i in range(256)}


#Returns the number of times each byte pair occurs
def get_count (tokens_dict): ##{ (32,100,97,121) : 3, (111,101): 1, ...}
    count_dict = defaultdict(int)

    for token_tuple, cnt in tokens_dict.items():
        for byte1, byte2 in zip(token_tuple, token_tuple[1:]):
            count_dict[(byte1,byte2)] += cnt
    
    return count_dict




def merge(tokens_dict, max_pair, replacement): 
    ##{ (32,100,97,121) : 3, (111,101): 1, ...} , max_pair:(100, 97), replacement: 257
    ## will return { (32,257,121): 3, (111, 101):1, ...}
    new_tokens_dict = {} 


    for ii, (token, count) in enumerate(tokens_dict.items()):
        new_token = []
        i = 0
        of_interest = False
        while i < len(token):
            if ( (i < len(token) - 1) and ( (token[i], token[i + 1]) == max_pair)):
                new_token.append(replacement)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        new_tokens_dict[tuple(new_token)] = count

    

    return new_tokens_dict


def split_based_on_specialtokens (special_tokens, text, incl_special_token=False):
    pattern = '|'.join ([re.escape(spl_tok) for spl_tok in special_tokens]) #escapes each special token and concatenates them with |
    if incl_special_token :
        pattern = "(" + pattern + ")"
    texts = re.split(pattern, text) #splits based on any matching special token (the pattern formed above)
    #"the sky is blue <|endoftext|> rain in spain" becomes ["the sky is blue ", " rain in spain"] <-- if incl_special_token is False
    return texts

def pretokenize(text, token_counts, gpt2_pat, special_tokens):

    texts = split_based_on_specialtokens (special_tokens, text)

    #Pre-tokenization on each piece of text until specialtoken
    for text in texts:  
        for match in re.finditer(gpt2_pat, text):
            pretoken = match.group()
            numerical_pretoken = tuple(pretoken.encode("utf-8"))
            token_counts[numerical_pretoken] = token_counts.get(numerical_pretoken, 0) + 1


def pretokenize_file(filepath, token_counts, gpt2_pat, special_tokens):

    #Read the file:
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    pretokenize(text, token_counts, gpt2_pat, special_tokens)

    



def tokenize (filepath, gpt2_pat, vocab_size, special_tokens):

    vocab = init_vocab()

    merges = {} # { (121,77): 256, (256,131): 257, ...}
    token_counts = {} #{ (32,100,97,121) : 3, (111,101): 1, ...}

    pretokenize_file (filepath, token_counts, gpt2_pat, special_tokens)


    #We add the special tokens at the end!
    num_epochs = vocab_size - (len(vocab) + len(special_tokens))

    for epoch in range(num_epochs):
        if (epoch % 100) == 0:
            print (f"On epoch {epoch}")

        #Lets find the most common pair first
        pair_counts = get_count(token_counts)

        pair_counts_sorted = {k: v for k, v in sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)}


        #print (f"pair counts = {pair_counts}")
        max_pair, max_cnt =  max(pair_counts.items(), key=lambda item: (item[1], item[0]))

        

        #Having found the most common pair, our goal now is to add the new word to
        #the vocab and merges, and also merge it in token_counts
        replacement = len(vocab)
        #print (f"max-pair = {max_pair} and replacement = {replacement}")
        merges[max_pair] = replacement

        #for int_byte in max_pair:
        #    vocab[replacement] = vocab.get(replacement, b'') + vocab[int_byte]
        #More elegant version of the above
        vocab[replacement] = b''.join([vocab[i] for i in max_pair])
            
        #print (f"pre merge {token_counts}")
        token_counts = merge(token_counts, max_pair, replacement) 
        #print (f"post merge {token_counts}")
        #print ("xx"*20)

        for k,v in pair_counts.items():
            assert (v >= 0)
        
        #merged_val = vocab[replacement].decode("utf-8")
        #print (f"merging to have formed {merged_val}")
  

    #Adding special tokens to vocab!
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    #merges = [ele[0] for ele in merges.items()] #In case a list is warranted as return. Revisit

    return vocab, merges


if __name__ == '__main__':

    filepath = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 5000

    gpt2_pat = settings.gpt2_pat
    special_tokens = settings.special_tokens

   

    #From different tokenizer attempts
    all_vocabs = []
    all_merges = []

    
    start = time.time()
    vocab, merges = tokenize (filepath, gpt2_pat, vocab_size, special_tokens)
    end = time.time()
    all_vocabs.append(vocab)
    all_merges.append(merges)
    print(f" took {end - start:.4f} seconds")

    #filehandler.serializedWrite(vocab, "vocab_tiny_5000.pkl")
    #filehandler.serializedWrite(merges, "merges_tiny_5000.pkl")
    filehandler.textWrite(vocab, "vocab_tiny_5000.txt")

    
    