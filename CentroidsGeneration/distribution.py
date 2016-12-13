from utilities import load_stf
import numpy as np
from scipy.spatial.distance import cosine
import time

#vsm = load_stf('glove.840B.300d.sample.txt',300)
#csm = np.load('centroids').item()
#distrib = np.zeros((100000,10))
#oFile = open('f_distrib','w+')

def dot_product(v1,v2):
	total = 0
	if len(v1) != len(v2):
		throw	
	for i in range(len(v1)):
		total += float(v1[i])*float(v2[i])
	return total

def centroid(vsm,w,k):
	total = np.zeros(len(vsm.word_vectors[vsm.dictionary[w]]))
	for v in vsm.most_similar(w,k+1):
		total += vsm.word_vectors[vsm.dictionary[v[0]]]
	total /= k
	return total
		
def lcent_similarity(w1,w2,vsm,gamma,k,c):
	v1 = vsm.word_vectors[vsm.dictionary[w1]]
	v2 = vsm.word_vectors[vsm.dictionary[w2]]
	v1v2 = dot_product(v1,v2)
	v1c = dot_product(v1,c)
	v1cg = np.power(v1c,gamma)
	return v1v2 - v1cg

def insert(v,sims,vec,val):
	nv = np.zeros(len(v))
	nsims = np.zeros((len(sims),300))
	swap = 0
	for i in range(len(v)):
		if v[i]<val:
			swap = 1
			break
	if swap == 0:
		return (v,sims)
	nv[:i] = v[:i]
	nsims[:i] = sims[:i]
	nv[i] = val
	nsims[i] = vec
	nv[i+1:] = v[i:len(v)-1]
	nsims[i+1:] = sims[i:len(sims)-1]
	return (nv,nsims)

def most_similar_lcent(vsm,csm,word,k,gamma):
	sims = np.zeros(10)
	vecs = np.zeros(10) 
	c = csm[word]
	for i,d_word in enumerate(vsm.dictionary):
		sim = lcent_similarity(word,d_word,vsm,gamma,k,c)
		(sims,vecs) = insert(vecs,sims,vsm.dictionary[d_word],sim)
	ret = []
	for i in range(10):
		ret.append((sims[i],vecs[i]))
	return ret
	
'''
centroids = {}
for i,j in enumerate(vsm.dictionary):
	if i%100 == 0:
		print i
	centroids[j] = centroid(vsm,j,11)

'''


#c = time.time()
#for j,w in enumerate(vsm.dictionary):
#    print j
#    print time.time() - c
#    c = time.time()
#    ms = most_similar_lcent(vsm,csm,w,11,2)
#    for k,s in enumerate(ms):
#	print s
#        i = vsm.dictionary[s]
#        distrib[i,k] += 1


#for c in centroids:
#    oFile.write(str(c) + u' ')
#    for i in centroids[c]:
#        oFile.write(str(i) + u' ')
#    oFile.write(u'\n')
#np.save(oFile,distrib)
#oFile.close()

