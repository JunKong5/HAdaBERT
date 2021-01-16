# HAdaBERT
This is code for Hierarchical BERT with an Adaptive Fine-Tuning Strategy for Document Classification

# Requirements
transformers == 3.1.0  
torch == 1.2.0  
python == 3.6.9  
easydict == 1.9  

# usage
- Datasets  
the Reuters, AAPD，IMDB and Yelp-2013 datasets are available at [here](https://git.uwaterloo.ca/jimmylin/hedwig-data/-/tree/master/datasets)

- Running the model over IMDB datasets  
adabert_encoder:  
python run.py --run train --dataset IMDB --mode adabert_encoder --version fine-tune  
global_encoder:  
python run.py --run train --dataset IMDB --mode gencoder --version global_encoder

