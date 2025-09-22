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

def update_pair_counts(pair_counts, token, count, replacement, max_pair):
    for ii in range(len(token)):
        byte1 = token[ii]
        if byte1 == replacement and ii > 0: #first term has no prefix!
            prefix = token[ii-1]
                        
            if prefix != replacement: #prev iteration already  accounted for this!                
                pair_counts[(prefix, replacement)] += count #b
                pair_counts[(prefix,max_pair[0])]  -= count #c       

        if (byte1 == replacement) and (ii < len(token) - 1): #not the last term
            suffix = token [ii+1]
            pair_counts[(replacement, suffix)] += count #d
            if suffix == replacement:
                suffix = max_pair[0]
            pair_counts[(max_pair[1], suffix)] -= count #e


def merge(tokens_dict, max_pair, replacement, in_place_update, pair_counts): 
    ##{ (32,100,97,121) : 3, (111,101): 1, ...} , max_pair:(100, 97), replacement: 257
    ## will return { (32,257,121): 3, (111, 101):1, ...}
    new_tokens_dict = {} 

    #pretokens_of_interest = [ ( (121,101,123),2) ,  ( (21,1,13),3) ... ]
    pretokens_of_interest = []

    for ii, (token, count) in enumerate(tokens_dict.items()):
        new_token = []
        i = 0
        of_interest = False
        while i < len(token):
            if ( (i < len(token) - 1) and ( (token[i], token[i + 1]) == max_pair)):
                new_token.append(replacement)
                of_interest = True
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        new_tokens_dict[tuple(new_token)] = count
        if of_interest:
            pretokens_of_interest.append ( (tuple(new_token), count))
    
    if (in_place_update):
        #print (f"updated_pretokens. = {updated_pretokens}")
        
        #Lets update the pair-counts based on updated_pretokens!
        #The five update rules are:
        #a. the max_pair is set to 0  
        #b. the prefix-[replacement] is added to pair-counts
        #c. the prefix-[max-pair-byte1] is subtracted from pair-counts!
        #d. [replacement]-suffix is added to pair-counts
        #e. [max-pair-byte2]-suffix is subtracted from pair-counts
        # for b and c, d and e we only look at those pretokens that warrant an update.
        
        
        #go through each pretoken_of_interest and update pair_cnt accordingly
        for  pretoken_of_interest in pretokens_of_interest: #( (111,110,101), 2)
            #print (f"pti = {pretoken_of_interest}")
            update_pair_counts(pair_counts, pretoken_of_interest[0], pretoken_of_interest[1], replacement, max_pair)         
        
        pair_counts[max_pair] = 0 #a

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

    

def worker(q, filepath, start, end, gpt2_pat, special_tokens):
    with open(filepath, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore") 
    
    '''
    pattern = '|'.join ([re.escape(spl_tok) for spl_tok in special_tokens]) #escapes each special token and concatenates them with |
    texts = re.split(pattern, text) #splits based on any matching special token 
    #"the sky is blue <|endoftext|> rain in spain becomes ["the sky is blue ", " rain in spain"]"
    '''
    texts = split_based_on_specialtokens (special_tokens, text)

    token_counts = defaultdict(int)
    for text in texts: #Pretokenization on each piece of text until specialtoken
        for match in re.finditer(gpt2_pat, text):
            pretoken = match.group()
            numerical_pretoken = tuple(pretoken.encode("utf-8"))
            token_counts[numerical_pretoken] += 1
    
    q.put(token_counts)


def parallel_pretokenize(filepath, token_counts, gpt2_pat, special_tokens):

    num_processes = 8
    processes = []
    q = Queue()

    #Below file-read from stanford 336
    with open(filepath, "rb") as f:
        boundaries = chunker.find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
    
    num_processes = min(num_processes, len(boundaries))

    for pnum, (start, end) in enumerate (zip(boundaries[:-1], boundaries[1:])):
        p = Process(target=worker, args=(q, filepath,start,end, gpt2_pat, special_tokens))
        processes.append(p)

    for pnum, process in enumerate(processes):
        print(f"starting process {pnum}")
        process.start()
    
    # Read all messages from the queue (unblocks child processes)
    items = [q.get() for _ in range(num_processes)]
    
    #barrier -- multiprocess hygiene to wait for each child to finish!
    for pnum, process in enumerate(processes):
        process.join()
    
    #Lets go through each item in items and return aggregate token_counts
    for item in items:
        for k,v in item.items():
            token_counts[k] = token_counts.get(k,0) + v
        
    #print (f"token counts = {token_counts}")


def tokenize (filepath, gpt2_pat, vocab_size, special_tokens, in_place_update, is_parallel):

    vocab = init_vocab()

    merges = {} # { (121,77): 256, (256,131): 257, ...}
    token_counts = {} #{ (32,100,97,121) : 3, (111,101): 1, ...}

    if is_parallel:
        parallel_pretokenize(filepath, token_counts, gpt2_pat, special_tokens)
    else:
        pretokenize_file (filepath, token_counts, gpt2_pat, special_tokens)


    #We add the special tokens at the end!
    num_epochs = vocab_size - (len(vocab) + len(special_tokens))

    for epoch in range(num_epochs):
        if (epoch % 100) == 0:
            print (f"On epoch {epoch}")

        #Lets find the most common pair first
        if (epoch == 0) or (in_place_update == False): #if in-place, pair_counts already updated!
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
        token_counts = merge(token_counts, max_pair, replacement, in_place_update, pair_counts) 
        #print (f"post merge {token_counts}")
        #print ("xx"*20)

        for k,v in pair_counts.items():
            assert (v >= 0)
    

    #Adding special tokens to vocab!
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    #print (f"merges = {merges}")
    #merges = [ele[0] for ele in merges.items()] #In case a list is warranted as return. Revisit

    #print (f"vocab = {vocab}")
    return vocab, merges


if __name__ == '__main__':

    filepath = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 8000

    gpt2_pat = settings.gpt2_pat
    special_tokens = settings.special_tokens

    in_place_update = True
    is_parallel = True

    #From different tokenizer attempts
    all_vocabs = []
    all_merges = []

    
    print (f"Starting in-place = {not in_place_update}  parallel = { is_parallel}")
    start = time.time()
    vocab, merges = tokenize (filepath, gpt2_pat, vocab_size, special_tokens, not in_place_update,  is_parallel)
    end = time.time()
    all_vocabs.append(vocab)
    all_merges.append(merges)
    print(f"in-place = {not in_place_update}  parallel = { is_parallel} took {end - start:.4f} seconds")

    
    print (f"Starting in-place = { in_place_update}  parallel = {not is_parallel}")
    start2 = time.time()
    vocab2, merges2 = tokenize (filepath, gpt2_pat, vocab_size, special_tokens,  in_place_update, not is_parallel)
    end2 = time.time()
    all_vocabs.append(vocab2)
    all_merges.append(merges2)
    print(f"in-place = { in_place_update}  parallel = {not is_parallel} took {end2 - start2:.4f} seconds")
    
    
    print (f"Starting in-place = {in_place_update}  parallel = {is_parallel}")
    start3 = time.time()
    vocab3, merges3 = tokenize (filepath, gpt2_pat, vocab_size, special_tokens,  in_place_update,  is_parallel)
    end3 = time.time()
    all_vocabs.append(vocab3)
    all_merges.append(merges3)
    print(f"in-place = { in_place_update}  parallel = { is_parallel} took {end3 - start3:.4f} seconds")

    print (f"Starting in-place = {not in_place_update}  parallel = {not is_parallel}")
    start4 = time.time()
    vocab4, merges4 = tokenize (filepath, gpt2_pat, vocab_size, special_tokens, not in_place_update, not  is_parallel)
    end4 = time.time()
    all_vocabs.append(vocab4)
    all_merges.append(merges4)
    print(f"in-place = {not in_place_update}  parallel = {not is_parallel} took {end4 - start4:.4f} seconds")

    print("--------"*30)
    
    #filehandler.serializedWrite(vocab3, "vocab.pkl")
    #filehandler.serializedWrite(merges3, "merges.pkl")

    #Checks
    assert len(all_merges) == len(all_vocabs)
    if len(all_vocabs) > 1:
        for _vocab1, _vocab2, _merges1, _merges2 in zip(all_vocabs, all_vocabs[1:], all_merges, all_merges[1:]):
            assert (_vocab1 == _vocab2)
            assert (_merges1 == _merges2)