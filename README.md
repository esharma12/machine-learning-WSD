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
