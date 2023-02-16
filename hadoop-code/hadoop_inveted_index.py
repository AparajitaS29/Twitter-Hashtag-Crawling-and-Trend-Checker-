import re #Regular Expression
from mrjob.job import MRJob
import ast
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import collections

class Count(MRJob):
    
    def mapper(self, _, line):
        data = ast.literal_eval(line)
        string = data['Tweet']
        string = re.sub(r'http\S+', '', string) #Code to remove URLs from the text

        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        answer = ''.join(filter(whitelist.__contains__, string))
        stop_words = set(stopwords.words('english'))
 
        word_tokens = word_tokenize(answer)
 
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
 
        filtered_sentence = []
 
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)

        answer = (" ").join(filtered_sentence)

        for word in answer.split(" "):
            yield(word.lower(),string)
             
         

    def reducer(self, word, listlines):
        counts = collections.Counter(list(listlines))
        new_list = sorted(listlines, key=lambda x: -counts[x])
        skinny = list(dict.fromkeys(new_list))
        yield (word, skinny)
     
if __name__ == '__main__':
    Count.run()