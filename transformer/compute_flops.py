#Q,k,v,o: [clen*dm^2] * 4 * L  
#Qkt V : [clen^2 * dm] * 2 * L
#FFN [up + down] : [clen *dm * dvocab] * 2 *L
#Final FFN : [clen * dm * dvocab]


clen = 1024
dm = 768 #1600
layers = 12#48
dvocab = 50257
dff = 4*dm


qkvo = clen*dm*dm * 4 * layers * 2 
qktv = clen*clen * dm * 2 * layers * 2
ffn_up_down = clen*dm*dff * 2 * layers * 2
final_ffn = clen*dm*dvocab *2

#the last two on top is to account for adds and mults
#A[M,N] * B[N,P] gives matrix of size A[M,P] with 2MNP adds+mult ops!

total =(qkvo + qktv + ffn_up_down + final_ffn)
print (f"total flop = {total} bn")
print ("Individual")
print (f"qkvo = {qkvo*100./total}")
print (f"qktv = {qktv*100./total}")
print (f"ffn_up_down = {ffn_up_down*100./total}")
print (f"final_ffn = {final_ffn*100./total}")
print ("---"*40)


clen = 1024 #16384 #See that as context len increases [q,k,t,v --> attention starts to consume more and more flops!]
dm = 1280 #1600
layers = 36 #48
dvocab = 50257
dff = 4*dm
#if it becomes ~=dff size then it starts to consume as many flops as the FFN


qkvo = clen*dm*dm * 4 * layers * 2 
qktv = clen*clen * dm * 2 * layers * 2
ffn_up_down = clen*dm*dff * 2 * layers * 2
final_ffn = clen*dm*dvocab *2


total =(qkvo + qktv + ffn_up_down + final_ffn)
print (f"total flop = {total} bn")
print ("Individual")
print (f"qkvo = {qkvo*100./total}")
print (f"qktv = {qktv*100./total}")
print (f"ffn_up_down = {ffn_up_down*100./total}")
print (f"final_ffn = {final_ffn*100./total}")
print (f"total = {total}")