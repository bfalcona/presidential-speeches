# Topic Modeling Presidential Speeches

This project analyzes the speech patterns (particularly word usage) of recent U.S. presidents from the Republican Party in order to recognize trends in policy changes within the Party as a whole.



The project makes extensive use of the [Natural Language Toolkit (NLTK)](https://www.nltk.org/), a library including the many linguistics-related functions used throughout. These functions include:
* ***Tokenization*** - the process of splitting a string into a list of, in this case, separate words and punctuation
  * For example, the string `'Hello, world!'` would be *tokenized* as `{'Hello', ',', 'world', '!'}`
* ***Lemmatization*** - the process of labeling inflected word forms with each *lemma* form (i.e. "dictionary form")
  * For example, the sentence "The dogs are barking at Mary's cats" would be *lemmatized* as "The dog be bark at Mary cat", making the sentence easier to analyze.
* ***Topic modeling*** - a dimensionality reduction technique whereby individual words are probabilistically grouped together into *topics* based on those words' tendency to appear in together in the same documents (i.e. based on the "topic" of the document)

Speech data was obtained from The Grammar Lab, an online corpus of speeches by U.S. presidents. This data can be located in the `presidential_speeches` folder. Several speeches were discarded as irrelevant for the project; these can be found in the `irrelevant_speeches` folder.

A .pdf version of the Python code is included for ease of reading. The original .ipynb (Jupyter Notebook) is also included.

This project was a collaboration between Brandon Falcona and Noah Pang as part of the spring 2020 section of the LIN 350 ("Analyzing Linguistic Data") course at The University of Texas at Austin. Over the course of the project, three reports were made including various details about the project's concepts and development. These can be found in the "Project Reports" folder.

Acknowledgments go to Dr. Katrin Erk for teaching our LIN 350 course and for her guidance over the this project's development.
