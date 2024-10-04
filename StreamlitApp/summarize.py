import nltk
import heapq
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class Preprocess:
    def _init_(self):
        pass

    def toLower(self, text):
        '''Converts the input text to lowercase'''
        return text.lower()

    def sentenceTokenize(self, text):
        '''Tokenizes the text into sentences'''
        return nltk.sent_tokenize(text)

    def preprocess_sentences(self, sentences):
        '''Tokenizes sentences, removes punctuation, stopwords, and lemmatizes'''
        word_tokenizer = nltk.RegexpTokenizer(r"\w+")
        special_characters = re.compile("[^A-Za-z0-9 ]")
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        processed_sentences = []
        for sentence in sentences:
            sentence = re.sub(special_characters, " ", sentence)
            words = word_tokenizer.tokenize(sentence)
            words = [word for word in words if word.lower() not in stop_words]
            words = [lemmatizer.lemmatize(word, pos='v') for word in words]
            processed_sentences.append(words)

        return processed_sentences

    def complete_preprocess(self, text):
        '''Performs complete preprocessing on input text'''
        text = self.toLower(text)
        sentences = self.sentenceTokenize(text)
        return self.preprocess_sentences(sentences)


class KeywordExtraction:
    def extract_keywords(self, text, top_n=10):
        '''Extracts top n keywords using TF-IDF'''
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')  # Adjusted min_df
        X = vectorizer.fit_transform([text])
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]

        top_n_keywords = feature_array[tfidf_sorting][:top_n]
        return top_n_keywords


class TopicModeling:
    def extract_topics(self, text, num_topics=3):
        '''Performs topic modeling using LDA'''
        vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')  # Adjusted min_df
        X = vectorizer.fit_transform([text])
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(X)

        topics = {}
        for i, topic in enumerate(lda_model.components_):
            topics[f"Topic {i+1}"] = [vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]]
        return topics


class NewsSummarization:
    def _init_(self):
        pass

    def extractive_summary(self, text, sentence_len=8, num_sentences=3):
        '''Generates an extractive summary based on sentence scoring'''
        preprocessor = Preprocess()
        tokenized_sentences = preprocessor.complete_preprocess(text)
        word_frequencies = {}

        # Calculate word frequencies
        for sentence in tokenized_sentences:
            for word in sentence:
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        maximum_frequency = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

        sentence_scores = {}
        sentence_list = sent_tokenize(text)
        for sent in sentence_list:
            for word in word_tokenize(sent.lower()):
                if word in word_frequencies:
                    if len(sent.split()) > sentence_len:
                        sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

        summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)
        return summary

    def get_rouge_score(self, reference_summary, generated_summary):
        '''Calculates ROUGE score for evaluating summaries'''
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_summary, generated_summary)
        return scores

    def evaluate_extractive(self, dataset, metric):
        '''Evaluates extractive summarization using a dataset'''
        summaries = [self.extractive_summary(text) for text in dataset["article"]]
        score = metric.compute(predictions=summaries, references=dataset["highlights"])
        return score

    def evaluate_abstractive(self, dataset, metric, summarizer):
        '''Evaluates abstractive summarization using a dataset'''
        summaries = [summarizer(text, max_length=120, min_length=80, do_sample=False)[0]['summary_text'] for text in dataset["article"]]
        score = metric.compute(predictions=summaries, references=dataset["highlights"])
        return score