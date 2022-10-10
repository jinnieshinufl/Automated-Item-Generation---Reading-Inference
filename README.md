# Automated-Item-Generation-Reading-Inference (Shin & Gierl, under review)

## Get Started
### Prerequisites
This code is prepared in python 3. A few python packages are required in order to run the code.
```
numpy == 1.19.5
nltk == 3.7.0
sklearn == 1.0.2
vaderSentiment == 3.3.2
pandas == 1.2.2
```
### Components 
You can find the two main folders ```main``` and ```data```

``Data``: This includes *example.txt* which provides the example data structure (corpus) in order to run the model.  
```Result```: The example generated question based on the `main.py` result
- ```main.py```: Includes the main model for training.   
- ```rule-based model.py```: Item model examples - coherent and divergent items 
- ```pre_processing.py```: Data preprocessing and cleaning (1)
- ```get_text.py```: Data preprocessing and cleaning (2)

```
## References 
TBA
