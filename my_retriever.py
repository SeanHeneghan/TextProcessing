import math
import operator
class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting
        self.df = self.df()   

    # Method performing retrieval for specified query
    def forQuery(self, query):
        #Find document sizes for each weighting
        #the document sizes would be the length of the document vector if actually constructed 
        bin_size = {}
        tf_size = {}
        D,x = max(((p,q) for num in self.index.values() for p,q in num.items()))
        tfidf_size={}
        #index dictionary: 'term' -> {doc:tf}
        for term in self.index:
            for doc in self.index[term]:
                tf = (self.index[term][doc])
                tfidf = (tf * (math.log10(D/self.df[term]))) #using idf as log10(|D|/df)
                if doc in bin_size:
                    bin_size[doc] +=1
                    tf_size[doc] += tf**2
                    tfidf_size[doc] += tfidf**2
                else:
                    bin_size[doc] = 1
                    tf_size[doc] = tf**2
                    tfidf_size[doc] = tfidf**2
        #####################################################            
        if self.termWeighting == 'binary':
            qd = {} #numerator of the RHS of similarity equation
            for term in query:
                if term in self.index:
                    for doc in self.index[term]:
                        if doc in qd:
                            qd[doc]+=1
                        else:
                            qd[doc]=1
            #compute similarity with weighted terms using sim(q,d) equation from slides
            sim = {}
            for doc in qd:
                sim[doc] = qd[doc]/math.sqrt(bin_size[doc])
                #sort to find the best results then grab the top ten
                sortedsim = sorted(sim.items(), key=operator.itemgetter(1))[-10:]
            #sorted sim returns dictionary with key:doc, value:similarity value
            #we just want the docs
            docids = []
            for doc in sortedsim:
                docids.append(doc[0])
            return docids
        #######################################################
        elif self.termWeighting == 'tf':
            qd={} #numerator of the RHS of similarity equation
            for term in query:
                if term in self.index:
                    for doc in self.index[term]:
                        weighted_query = query[term]
                        weighted_doc = self.index[term][doc] #term frequency
                        if doc in qd:
                            qd[doc]+= weighted_query*weighted_doc #q x d
                        else:
                            qd[doc]= weighted_query*weighted_doc
            #compute similarity with weighted terms using sim(q,d) equation from slides
            sim = {}
            for doc in qd:
                sim[doc]=qd[doc]/math.sqrt(tf_size[doc])
                #sort to find the best results then grab the top ten
                sortedsim = sorted(sim.items(), key=operator.itemgetter(1))[-10:]
            #sorted sim returns dictionary with key:doc, value:similarity value
            #we just want the docs
            docids = []
            for doc in sortedsim:
                docids.append(doc[0])
            return docids
        #########################################################
        elif self.termWeighting == 'tfidf':
            qd={} #numerator of the RHS of similarity equation
            for term in query:
                if term in self.index:
                    for doc in self.index[term]:
                        weighted_query = query[term]*(math.log10(D/self.df[term])) #q x idf
                        weighted_doc = (self.index[term][doc])*(math.log10(D/self.df[term])) #tf x idf
                        if doc in qd:
                            qd[doc]+= weighted_query*weighted_doc
                        else:
                            qd[doc]= weighted_query*weighted_doc
            #compute similarity with weighted terms using sim(q,d) equation from slides
            sim = {}
            for doc in qd:
                sim[doc] = qd[doc]/math.sqrt(tfidf_size[doc])
                #sort to find the best results then grab the top ten
                sortedsim = sorted(sim.items(), key=operator.itemgetter(1))[-10:]
            #sorted sim returns dictionary with key:doc, value:similarity value
            #we just want the docs
            docids = []
            for doc in sortedsim:
                docids.append(doc[0])
            return docids
    ###############################################################
    #document frequency so we can compute tfidf
    #stores each term from the index into a dictionary along with the 
    #number of documents it appears in.
    def df(self):
        df={}
        for term in self.index:
            for doc in self.index[term]:
                if term in df:
                    df[term]+=1
                else:
                    df[term]=1
        return df