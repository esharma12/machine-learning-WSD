# machine-learning-WSD
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
