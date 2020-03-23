import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()


    def file_to_string(path):
        with open(path) as file:
            text_as_list = [line for line in file]
            index_end = None
            for line in text_as_list:
                if "X-FileName:" in line:
                    index_begin = text_as_list.index(line) + 1
                if "Original Message" in line:
                    index_end = text_as_list.index(line)
            if not index_end:
                index_end = len(text_as_list) + 1
        text_string = " ".join(line.strip() for line in text_as_list[index_begin: index_end])
        return text_string



    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    file_dict = {}
    texts_label_total = []
    for folder in os.listdir(args.inputdir):
        filepath = os.path.join(args.inputdir,folder)
        if os.path.isdir(filepath):
            files_texts_list = []
            for file in os.listdir(filepath):
                text = file_to_string(filepath + "/" + file)
                files_texts_list.append(text)
                texts_label_total.append((text, folder))
            file_dict[folder] = files_texts_list


    texts_total, labels_total = [texts for texts, labels in texts_label_total], [labels for texts, labels in
                                                                                 texts_label_total]

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    vectorizer = TfidfVectorizer()
    print("---Features Extracting Applying TfidVectorizer---")
    X = vectorizer.fit_transform(texts_total)
    print("Size of TFIDF MATRIX: ", X.shape)
    print()
    svd = TruncatedSVD(n_components=args.dims)
    print("---Dimension Reducting Applying TruncatedSVD---")
    X = svd.fit_transform(X)
    print("Reduced Size of MATRIX: ", X.shape)
    print()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, labels_total, test_size=args.testsize/100, shuffle=True)
    table = np.zeros((X.shape[0], X.shape[1] + 2))
    author_index_map = dict([(b, a) for a, b in list(enumerate(set(labels_total)))])
    train_test_map = {"train": 0, "test": 1}
    columns = [str(i) for i in range(1, args.dims + 1)]
    columns.append("author")
    columns.append("test_train")
    table = pd.DataFrame(table, columns=columns)
    table.iloc[:len(Xtrain), 0:args.dims] = Xtrain
    y_train_index = [author_index_map[author] for author in ytrain]
    y_test_index = [author_index_map[author] for author in ytest]
    table.iloc[:len(Xtrain), -2] = y_train_index
    table.iloc[:len(Xtrain), -1] = 0
    table.iloc[len(Xtrain):, 0:args.dims] = Xtest
    table.iloc[len(Xtrain):, -2] = y_test_index 
    table.iloc[len(Xtrain):, -1] = 1
    print("The Output is Purely Numeric")
    print("Autor-Index-Map:")
    print(author_index_map)
    print("Train-Test-Index-Map")
    print(train_test_map)
    

    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    table.to_csv(args.outputfile)
    print("Done!")
    
