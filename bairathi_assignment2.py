import sys
import nltk 
from nltk.corpus import udhr
from nltk import word_tokenize
from nltk import bigrams
from nltk.util import ngrams
from nltk import FreqDist
from nltk import ConditionalFreqDist
import re

'''
	Class LanguageModel: Calculates unigram, bigram and trigram probability for the given words
'''
class LanguageModel:
	# Initialize the Model with the given UDHR file
	def __init__(self, fileid):
		try:
			# Reads the UDHR file
			corpus = udhr.raw(fileid)
		except:
			print("UDHR language file " + fileid + " does not exist", file=sys.stderr)
			sys.exit(1)

		# Generate training dataset, lowercase and newlines converted to space
		self.train = re.sub(r'[\n]+', ' ', corpus[0:1000].strip().lower())
		# Generate dev dataset
		self.dev = corpus[1000:1100]

		# Convert training words to single characters
		tokens = list(self.train)
		self.unigram = tokens
		self.bigram = list(nltk.bigrams(tokens))
		self.trigram = list(nltk.trigrams(tokens))
		# Generate unigram frequency distirbution
		self.unigramFreq = FreqDist(self.unigram)
		# Generate bigram frequency distribution
		self.bigramFreq = ConditionalFreqDist(self.bigram)
		# Generate trigram frequency distribution
		self.trigramFreq = ConditionalFreqDist(list(((w0, w1), w2) for w0, w1, w2 in self.trigram))

	# Calculate unigram probability for the given word
	def calculate_unigram_probability(self, word):
		word = word.strip().lower()
		chars = list(word)
		probability = 1

		# Calculate probability of individual characters using prepared Frequency Distribution
		for char in chars:
			probability = probability * self.unigramFreq.freq(char)
		return probability

	# Calculate bigram probability for the given word
	def calculate_bigram_probability(self, word):
		word = word.strip().lower()
		chars = list(word)
		probability = 1

		# Calculate probability of individual characters using prepared Conditional Frequency Distribution
		for i, char in enumerate(chars):
			if i == 0:
				continue
			probability = probability * self.bigramFreq[chars[i - 1]].freq(char)
		return probability

	# Calculate trigram probability for the given word
	def calculate_trigram_probability(self, word):
		word = word.strip().lower()
		chars = list(word)
		probability = 1

		# Calculate probability of individual characters using prepared Conditional Frequency Distribution
		for i, char in enumerate(chars):
			if i <= 1:
				continue
			probability = probability * self.trigramFreq[(chars[i - 2], chars[i - 1])].freq(char)
		return probability

'''
	Read words from UDHR documents, outputs probabilities of each word and accuracy of the model
	Parameters:
		modelFile: The fileid of file from 'UDHR' package on which LanguageModel must be trained
		modelLanguage: The name of the language of modelFile. It is only used for the output purpose
		dataFile: The fileid of file from 'UDHR' package from which words must be read and tested on the LanguageModel
		dataLanguage: The name of the language of dataFile. It is only used for the output purpose.
'''
def perform_experiment(modelFile, modelLanguage, dataFile, dataLanguage):
	languageModel = LanguageModel(modelFile)
	try:
		# Read test words
		words = udhr.words(dataFile)[0:1000]
	except:
		print("UDHR language file " + dataFile + " does not exist", file=sys.stderr)
		sys.exit(1)

	# All words in the test set
	countWords = len(words)
	# Words successfully predicted by unigram model
	unigramPredicted = 0
	# Words successfully predicted by bigram model
	bigramPredicted = 0
	# Words successfully predicted by trigram model
	trigramPredicted = 0

	print("\n# Model: " + modelLanguage + ", Test Dataset: " + dataLanguage)
	print("+----------------------+---------------------+---------------------+---------------------+")
	print("| Word                 | Unigram Probability | Bigram Probability  | Trigram Probability |")
	print("|----------------------|---------------------|---------------------|---------------------|")
	for word in words:
		unigramProbability = languageModel.calculate_unigram_probability(word)
		if(unigramProbability > 0):
			unigramPredicted = unigramPredicted + 1

		bigramProbability = languageModel.calculate_bigram_probability(word)
		if(bigramProbability > 0):
			bigramPredicted = bigramPredicted + 1

		trigramProbability = languageModel.calculate_trigram_probability(word)
		if(trigramProbability > 0):
			trigramPredicted = trigramPredicted + 1

		print("| %20s | %19.17f | %19.17f | %19.17f |" % (word, unigramProbability, bigramProbability, trigramProbability) )
	print("|----------------------|---------------------|---------------------|---------------------|")
	print("| %20s | %18.5f%% | %18.5f%% | %18.5f%% |" % ("Accuracy", unigramPredicted * 100 / countWords, bigramPredicted * 100 / countWords, trigramPredicted * 100 / countWords))
	print("+----------------------+---------------------+---------------------+---------------------+")

# Question: 1.
# Language prediction of English words using English language Model
perform_experiment('English-Latin1', 'English', 'English-Latin1', 'English')
# Language prediction of French words using English Language Model
perform_experiment('English-Latin1', 'English', 'French_Francais-Latin1', 'French')

# Question: 2.
# Language prediction of Spanish words using Spanish Language Model
perform_experiment('Spanish_Espanol-Latin1', 'Spanish', 'Spanish_Espanol-Latin1', 'Spanish')
# Language prediction of Italian words using Spanish Language Model
perform_experiment('Spanish_Espanol-Latin1', 'Spanish', 'Italian_Italiano-Latin1', 'Italian')