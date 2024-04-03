'''
Esha Sharma 03/24/24 CMSC 416 PA4
Machine Learning Model Word Sense Disambiguation 

Naive Bayes Model Accuracy and Confusion Matrix:            Logistic Regression Model Accuracy and Confusion Matrix:            Stochastic Gradient Descent Model Accuracy and Confusion Matrix:
Accuracy is: 93.65079365079364%                             Accuracy is: 81.74603174603175%                                     Accuracy is: 79.36507936507937%

         phone  product                                                phone  product                                                    phone  product
                       
phone       69        5                                    phone       55        6                                              phone       48        2
product      3       49                                    product     17       48                                              product     24       52

In my PA3 program, my baseline accuracy was 42.9% and my overall accuracy of tagging was 81.75% In PA3, I used a ranked decision list with log-likelihood ratios. My confusion matrix from PA3 is below:
Confusion Matrix:
               phone    product


phone          64       15
product        8        39
The model with the highested accuracy is Multinomial Naive Bayes, with an accuracy of 93.65079365079364%. Multinomial Naive Bayes is a classifier that implements the naive Bayes algorithm. It works on the naive assumption that all features used
for classification are indepentent of each other. It is a machine learning that works well with discrete data like the data used in WSD and other text classification uses. It is a supervised-learning classification model that
represents data as word vectors with frequencies for each word. The conditional probabilities in Naive Bayes are calculated using these frequencies. The MN Naive Bayes model in sci-kit learn also uses laplace 
smoothing to avoid null probabilities. The next highest accuracy was both my decision list model in PA3 and the logistic regression model implemented in this program. Both models have an accuracy of around 81.75% which is very interesting.
My decision list model from PA3 used log-likelihood ratios from the bag of words features extracted from the training data. Logistic Regression is a Supervised Learning Classification linear model. In this model, the probabilities of a feature and sense is calculated
using a logistic function similar to PA3. It models the probabilities of a discrete or binary output given input of independent variables. The lowest accuracy is the Stochastic Gradient Descent Model with an accuracy of 79.36507936507937%. This is still very close to the accuracies 
of the last two modesl mentioned. The Stochastic Gradient Descent model is Supervised Learning Classicial linear machine learning model. This models implements a regular linear model but with an SGD, a gradient that searches for the model's optimum minimum and maximum value. Instead of using the whole
dataset for each iteration, it randomly picks a sample data, updating the model at each iteration, learning more accurately as it goes. 

1) The problem to be solved is to train a minimum of three different machine learning models from sci-kit learn that will perform word sense disambiguation, essentially that will assign a sense, 'phone' or 'product', to the word 'line' in a sentence. 
Essentially, the goal is to correctly ascertain the sense of a word using contexutal features in the training data itself. First, we have an ambiguous word with context, and after the word sense 
disambiguation algorithm is employed, then we can determine the sense of the ambiguous word. The training data consists of sentences/paragraphs of information of a specific context separated by <context> tags, and tagged with one of the two senses.
The features chosen for each sense for the machine learning models are bag of words, so all of the words in the training data associated with a specific sense, except for punctuation, digits, stopwords, and the tags witin the raw training data. 
A sci-kit learn feature extraction model, Dict Vectorizer, is used to created arrays of feature vectors from feature dictionaries. I chose my own three ml sci-kit-learn models to perform WSD, Multinomial Naive Bayes, Logistic Regression, and Stochastic Gradient Descent Clasifier. All three models 
are Supervised Learning Classification models. For each sci-kit learn model, the training feature vocab is fit and transformed to it. Essentially, the specific learns from the training vocab, and then transforms the feature vectors from the testing data using what it has learned
to make predictions of the senses of the word 'line/lines' for each context in the test data. 
2) Usage instructions: python3 wsd-ml.py line-train.txt line-test.txt [OPTIONAL: ml-model] > my-line-answers.txt
Example inputs for wsd-ml.py include: python3 wsd-ml.py line-train.txt line-test.txt NaiveBayes > my-line-answers.txt, or python3 wsd-ml.py line-train.txt line-test.txt > my-line-answers.txt. To use Logistic Regression, input LogisticRegression for ml-model. To use Stochastic Gradient Descent, input StochasticGD for ml-model. 
The program runs using python 3. line-train.txt line-test.txt are txt files that consist of the training and testing data, respectively. ml-model is the learning machine model that will be used to perform WSD, if no model is passed an argument, the program will default to Naive Bayes
my-line-answers.txt is where the output of answer tags consisting of the sense tagged to the pieces of testing text is outputted.
Examples of output when the usage instructions are inputted are the following:
For Naive Bayes (my-line-answers.txt):
<answer instance="line-n.w8_059:8174:" senseid="phone"/>
<answer instance="line-n.w7_098:12684:" senseid="phone"/>
<answer instance="line-n.w8_106:13309:" senseid="phone"/>...
For Logistic Regression (my-line-answers.txt):
<answer instance="line-n.w8_059:8174:" senseid="phone"/>
<answer instance="line-n.w7_098:12684:" senseid="phone"/>
<answer instance="line-n.w8_106:13309:" senseid="phone"/>...
For Stochastic Gradient Descent (my-line-answers.txt):
<answer instance="line-n.w8_059:8174:" senseid="phone"/>
<answer instance="line-n.w7_098:12684:" senseid="phone"/>
<answer instance="line-n.w8_106:13309:" senseid="phone"/>...
3) The algorithm I used to perform word sense disambiguation with three ml models from sci-kit learn is described by the following. First, I read in the file names and specific machine learning model from the command line. If no ml-model is inputted, I revert to default model: Naive Bayes.
In order to generalize the program such that any two senses can be used, I found all of the senses in the training data using regex, and then used the Counter module to count the number of each of the senses. This dictionary also allowed me to grab the specific 
senses and store them in strings. Then, I extract all of the features in the training data and separated them based on the sense they were associated with into two lists. In order to do this, I used regex to grab all the words (Bag of Words method) between every <context> and </context> tag. 
Once the features were extracted, I cleaned up the list of features in the main method using re modules where I removed all numbers and puncuation from the feature lists before I went on to remove stopwords from the lists.
To remove the stopwords, I used the nltk stopword, and I extended it to include the names of the tags I inadvertenly grabbed, like <s> and <p> and <context>, punctuation. Then, I used the Counter module to
calculate the frequency of sense1 feature vocabulary and sense2 feature vocabulary. I added the counter dictionaries for both senses to a final feature list, later to be used in Dict Vectorizer. Then, I read in the testing data, extracted all of the instance ids into a list, and all of the testing text/data into a list using the context tags.
I cleaned up the testing text by removing all digits, punctuations, all words in the stopword list. Then, I used Counter to create a feature vector for each untagged context in the testing data. I initialized a Dict Vectorizer, a feature extraction model from sci-kit learn, to fit and transform the feature vocabs associated with both senses, but separated,
from dicts with feature value pairs to array of feature vectors. This becomes X, or the independent variable that trains and fits the machine learning models. I also declared Y, or the dependent variable that will be predicted by the model, in this case, the senses 'phone' and 'product'. 
I marked 0 for sense1 and 1 for sense2 in Y. Then I enter if-elif statements that will perform WSD based on the ml-model specified by the command line. The ml-models are Multinomial Naive Bayes, Logistic Regression, 
and Stochastic Gradient Descent. For each model, it initializes the model, fits the X and Y variables to the model, and then predicts senses of the testing data by transforming testing data using already fitted and transformed Dict Vectorizer instance.
Then, uses the prediction array outputted by the model to determine the sense predicted for each test context, if prediction is 0, then sense1 was predicted, if prediction is 1, then sense2 was predicted. Then, the correctly formated answer tags with the associated instance id and predicted sense is 
outputted to my-line-answers.txt

'''
from collections import Counter
import sys
from sys import argv
import re
from nltk.corpus import stopwords 
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import numpy as np

def main():
    #save file names from command line
    train_file = str(argv[1])
    test_file = str(argv[2])
    model = ""
    if len(sys.argv) < 4:
        model = ""
    else:
        model = str(argv[3])

    #open and read test file
    file = open(train_file, 'r')
    train_data = file.read()
    train_text = ''.join(train_data)
    #find all senses and count frequencies
    senses = re.findall(r'senseid="(.*)"/>', train_text)
    counted_senses = dict(Counter(senses))
    #grab generalized senses strings and their frequences
    sense1, sense2 = counted_senses.keys()
 
    sense1_features = []
    sense2_features = []
    #use regex to grab all features between context tags
    train_contexts = re.findall(r'<context>\n(.*)\n</context>', train_text)
    for i in range(0, len(train_contexts)):
        #using stored senses for each context, sorts all training contexts
        if senses[i] == sense1:
            sense1_features.append(train_contexts[i])
        else:
            sense2_features.append(train_contexts[i])
    
    #download nltk stopword list and extend it to include tags in training data and puncuation
    stop_words = stopwords.words('english')
    stop_words.extend([".", ",", "s", "p", "context", "senseid", ";", "-", "--", "!", "?", "'", "'s", "head"])

    #remove puncuations and digits from feature list for sense1
    features1_string = re.sub(r'[0-9]', '', ' '.join(sense1_features).lower())
    sense1_features = re.findall(r"[\w']+|[.,!?;]", features1_string)
    features1_string = ' '.join(sense1_features)

    #remove puncuations and digits from feature list for sense2
    features2_string = re.sub(r'[0-9]', '', ' '.join(sense2_features).lower())
    sense2_features = re.findall(r"[\w']+|[.,!?;]", features2_string)
    features2_string = ' '.join(sense2_features)
   
    stopwords_dict = Counter(stop_words)
    #use list comprehension to remove stopwords from sense1 features
    features1_string = ' '.join([word for word in features1_string.split() if word not in stopwords_dict])
    sense1_features = features1_string.split()
    #use list comprehension to remove stopwords from sense2 features
    features2_string = ' '.join([word for word in features2_string.split() if word not in stopwords_dict])
    sense2_features = features2_string.split()

    #consolidate and count frequencies of training features for sense1 and sense2, append whole sense1 and sense2 feature strings to list, to later be used in modeling
    #features include the bag of words for each sense, so all words except for digits, punctuation, tags in un-tokenized training data, and stopwords
    #many features for each sense
    counted_sense1_features = Counter(sense1_features)
    counted_sense2_features = Counter(sense2_features)
    final_features = []
    final_features.append(counted_sense1_features)
    final_features.append(counted_sense2_features)

    f = open(test_file, 'r')
    test_data = f.read()
    test_text = ''.join(test_data)
    #extract all instance ids and save them into a list
    test_ids = re.findall(r'id="(.*):">', test_text)
    #extract all context test data and save into list, separated based on instance id
    test_contexts = re.findall(r'<context>\n(.*)\n</context>', test_text)

    for item in test_contexts:
        #clean context data, removing digits and puncutation and stopwords, and then reinstantiating the list with cleaned up string
        item_context_string = item.lower()
        item_context_string = re.sub(r'[0-9]', '', item_context_string)
        item_context = re.findall(r"[\w']+|[.,!?;]", item_context_string)
        item_context_string = ' '.join(item_context)
        item_context_string = ' '.join([word for word in item_context_string.split() if word not in stopwords_dict])
        counted_words = item_context_string.split()
        #obtain feature vector for each specific context using Counter 
        counted_words = Counter(counted_words)
        counted_words = dict(counted_words)
        i = test_contexts.index(item)
        test_contexts[i] = counted_words

    #initialize dictvectorizer which transforms the final_features lists into sense1 and sense2 vectors, X is the final
    #feature vocab, and Y is the feature senses, 0 marks sense1 and 1 marks sense2
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(final_features)
    Y = np.array([0, 1])
  
    #three machine models: Naive Bayes (Multinomial), Logistic Regression, and Stochastic Gradient Descent
    #default model is Naive Bayes, when no model is specified in command line
    #uses argument in command line to determine which machine learning model to perform the word sense disambiguation with
    if (model == "NaiveBayes") or (model == ""):
        #initialize classifier model for Multinomial Naive Bayes
        mnb = MultinomialNB()
        #fit feature vectors and senses to model
        mnb.fit(X, Y)
        #obtain prediction array by using already fitted transformer from dict vectorizer to transform test data
        array = mnb.predict(dv.transform(test_contexts)) 
        #if prediction is 0, then output answer tag with sense1, if prediction is 1, then output answer tag with sense2
        for i in range(0, len(array)):
            if array[i] == 0:
                print('<answer instance="' + test_ids[i] + ':" senseid="' + sense1 + '"/>')
            elif array[i] == 1:
                print('<answer instance="' + test_ids[i] + ':" senseid="' + sense2 + '"/>')
    elif model == "LogisticRegression":
        #initialize classifier model for Logistic Regression
        logreg = LogisticRegression()
        #fit feature vectors and senses to model
        logreg.fit(X, Y)
        #obtain prediction array by using already fitted transformer from dict vectorized training data to transform test data
        array = logreg.predict(dv.transform(test_contexts))
        #if prediction is 0, then output answer tag with sense1, if prediction is 1, then output answer tag with sense2
        for i in range(0, len(array)):
            if array[i] == 0:
                print('<answer instance="' + test_ids[i] + ':" senseid="' + sense1 + '"/>')
            elif array[i] == 1:
                print('<answer instance="' + test_ids[i] + ':" senseid="' + sense2 + '"/>')
    elif model == "StochasticGD":
        #initialize classifier model for Stochastic Gradient Descent
        clf = SGDClassifier()
        #feature feature vectors and senses to model
        clf.fit(X, Y)
        #obtain prediction array by using already fitted transformer from dict vectorized training data to transform test data
        array = clf.predict(dv.transform(test_contexts))
         #if prediction is 0, then output answer tag with sense1, if prediction is 1, then output answer tag with sense2
        for i in range(0, len(array)):
            if array[i] == 0:
                print('<answer instance="' + test_ids[i] + ':" senseid="' + sense1 + '"/>')
            elif array[i] == 1:
                print('<answer instance="' + test_ids[i] + ':" senseid="' + sense2 + '"/>')

if __name__ == "__main__":
    main()