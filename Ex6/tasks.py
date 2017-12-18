import numpy as np
from sklearn import svm
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score
from porterStemmer import porterStemmer
from getVocabList import getVocabList
import re


def gaussianKernel(x1, x2, sigma):
    """
    @brief      returns a radial basis function kernel between x1 and x2

    @param      x1     The x 1
    @param      x2     The x 2
    @param      sigma  The sigma

    @return     radial basis
    """
    return np.exp(-np.sum((x1-x2)**2/(2*sigma**2)))


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

# You need to return the following variables correctly.
    C = 1
    sigma = 0.3

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set.
#               You can use svmPredict to predict the labels on the cross
#               validation set. For example,
#                   predictions = svmPredict(model, Xval)
#               will return the predictions on the cross validation set.
#
#  Note: You can compute the prediction error using
#        mean(double(predictions ~= yval))
#
    best_score = 0

    for c in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for sigm in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
            gamma = 1.0 / (2.0 * sigm ** 2)
            clf = svm.SVC(C=c, gamma=gamma).fit(X, y)

            p = clf.predict(Xval)
            score = np.mean(np.double(p == yval))
            # score = clf.score(Xval, yval)
            # score = np.mean(cross_val_score(clf, Xval, yval))

            if best_score < score:
                best_score = score
                C = c
                sigma = sigm

# =========================================================================
    # print(best_score, c, sigma)
    return C, sigma


def processEmail(email_contents):
    """preprocesses a the body of an email and
    returns a list of word_indices
    word_indices = PROCESSEMAIL(email_contents) preprocesses
    the body of an email and returns a list of indices of the
    words contained in the email.
    """

# Load Vocabulary
    vocabList = getVocabList()

# Init return value
    word_indices = []

# ========================== Preprocess Email ===========================

# Find the Headers ( \n\n and remove )
# Uncomment the following lines if you are working with raw emails with the
# full headers

# hdrstart = strfind(email_contents, ([chr(10) chr(10)]))
# email_contents = email_contents(hdrstart(1):end)

# Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
# Looks for any expression that starts with < and ends with > and replace
# and does not have any < or > in the tag it with a space
    rx = re.compile('<[^<>]+>|\n')
    email_contents = rx.sub(' ', email_contents)
# Handle Numbers
# Look for one or more characters between 0-9
    rx = re.compile('[0-9]+')
    email_contents = rx.sub('number ', email_contents)

# Handle URLS
# Look for strings starting with http:// or https://
    rx = re.compile('(http|https)://[^\s]*')
    email_contents = rx.sub('httpaddr ', email_contents)

# Handle Email Addresses
# Look for strings with @ in the middle
    rx = re.compile('[^\s]+@[^\s]+')
    email_contents = rx.sub('emailaddr ', email_contents)

# Handle $ sign
    rx = re.compile('[$]+')
    email_contents = rx.sub('dollar ', email_contents)

# ========================== Tokenize Email ===========================

# Output the email to screen as well
    print('==== Processed Email ====')

# Process file
    l = 0

    # Remove any non alphanumeric characters
    rx = re.compile('[^a-zA-Z0-9 ]')
    email_contents = rx.sub('', email_contents).split()

    for content in email_contents:

        # Tokenize and also get rid of any punctuation
        # str = re.split('[' + re.escape(' @$/#.-:&*+=[]?!(){},''">_<#')
        #                                + chr(10) + chr(13) + ']', str)

        # content = word_tokenize(content)

        # Stem the word
        # (the porterStemmer sometimes has issues, so we use a try catch block)
        try:
            content = porterStemmer(content)
        except:
            content = ''
            continue

        # Skip the word if it is too short
        if len(content) < 1:
           continue

        # Look up the word in the dictionary and add to word_indices if
        # found
        # ====================== YOUR CODE HERE ======================
        # Instructions: Fill in this function to add the index of str to
        #               word_indices if it is in the vocabulary. At this point
        #               of the code, you have a stemmed word from the email in
        #               the variable str. You should look up str in the
        #               vocabulary list (vocabList). If a match exists, you
        #               should add the index of the word to the word_indices
        #               vector. Concretely, if str = 'action', then you should
        #               look up the vocabulary list to find where in vocabList
        #               'action' appears. For example, if vocabList{18} =
        #               'action', then, you should add 18 to the word_indices
        #               vector (e.g., word_indices = [word_indices  18] ).
        #
        # Note: vocabList{idx} returns a the word with index idx in the
        #       vocabulary list.
        #
        # Note: You can use strcmp(str1, str2) to compare two strings (str1 and
        #       str2). It will return 1 only if the two strings are equivalent.
        #

        # print([i for i, val in enumerate(content) if val in vocabList])
        try:
            word_indices.append(vocabList.index(content))
        except:
            continue

        # =============================================================

        # Print to screen, ensuring that the output lines are not too long
        if (l + len(content) + 1) > 78:
            print(content)
            l = 0
        else:
            print(content),
            l = l + len(content) + 1

# Print footer
    print('=========================')
    return word_indices


def emailFeatures(word_indices):
    """takes in a word_indices vector and
    produces a feature vector from the word indices.
    """

# Total number of words in the dictionary
    n = 1899

# You need to return the following variables correctly.
    x = np.zeros(n)
# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return a feature vector for the
#               given email (word_indices). To help make it easier to
#               process the emails, we have have already pre-processed each
#               email and converted each word in the email into an index in
#               a fixed dictionary (of 1899 words). The variable
#               word_indices contains the list of indices of the words
#               which occur in one email.
#
#               Concretely, if an email has the text:
#
#                  The quick brown fox jumped over the lazy dog.
#
#               Then, the word_indices vector for this text might look
#               like:
#
#                   60  100   33   44   10     53  60  58   5
#
#               where, we have mapped each word onto a number, for example:
#
#                   the   -- 60
#                   quick -- 100
#                   ...
#
#              (note: the above numbers are just an example and are not the
#               actual mappings).
#
#              Your task is take one such word_indices vector and construct
#              a binary feature vector that indicates whether a particular
#              word occurs in the email. That is, x(i) = 1 when word i
#              is present in the email. Concretely, if the word 'the' (say,
#              index 60) appears in the email, then x(60) = 1. The feature
#              vector should look like:
#
#              x = [ 0 0 0 0 1 0 0 0 ... 0 0 0 0 1 ... 0 0 0 1 0 ..]
#
#
    x[word_indices] = 1

# =========================================================================

    return x
