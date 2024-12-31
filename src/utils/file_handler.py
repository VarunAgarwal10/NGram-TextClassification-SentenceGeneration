"""
This module is used for reading the dataset and
doing basic preprocessing operations for cleaning
the dataset
"""


import re
import math

class FileManager:
    """
    A class for managing file operations - Read and Write
    """
    @staticmethod
    def read_file(file_path):
        """
        Method to read file from the file path
        """

        f = open(file_path, encoding="utf8")
        list_docs = f.readlines()
        f.close()
        return list_docs


class DocumentProcessor:
    """
    A class for handling the document preprocessing steps
    """

    @staticmethod
    def clean_punct(txt):
        """
        Method to clean the text by removing punctuations: ?,.!
        """
        pattern = re.compile(r'[^a-z\s?\.,!]+')
        result = pattern.sub('', txt)
        return result

    @staticmethod
    def preprocess_docs(list_docs):
        """
        Iterates over each document, removes punctuations and leading
        and trailing whitespaces and adds START and END tokens to the document
        """

        cleaned_docs = []
        for i in range(len(list_docs)):
            lower_doc = list_docs[i].lower().strip("\n ")
            lower_doc = lower_doc.strip()
            cleaned_doc = "<START> " + DocumentProcessor.clean_punct(lower_doc) + " <END>"
            cleaned_docs.append(cleaned_doc)

        return cleaned_docs


class TrainTestGenerator:
    """
    Creates the train and test docs from the documents
    """
    @staticmethod
    def split_docs(docs, train_ratio = 0.9):
        n_docs = len(docs)
        train_docs = docs[:math.ceil(n_docs*train_ratio)]
        test_docs = docs[math.ceil(n_docs*train_ratio)::]
        return train_docs,test_docs