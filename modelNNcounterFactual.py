"""
Using pytorch with Lending Club data for multivariate regression on 4 variables:
* y1: probability of defaulting on a loan
* y2: timing of default (conditional on above): observed and synthetic/counterfactual
* y3: probability of repaying a loan early
* y4: timing of early repayment (conditional on above) observed and synthetic/counterfactual

Timing, y2 and y4: as fraction of loan term (interpreted as conditional on event y1 or y3)
y1=y3=0 implies neither default, nor early repayment, in which case y2=y4=1 

#the idea is not to use dummy variables (neither hand coded nor pandas), as that approach
does not scale to large number of categories (within a variable).
 
The preferred approach in ML is to use vector embeddings instead for large categories (such as 
products in retail, or hotel rooms, or homes). So, pass in just encoded col for each cat variable, 
and then use an embedding layer (for each one). Adapted for multivariate regression from the 
example by https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/

all data must be numeric: can use sk-learn's label encoder or Spark's stringencoder 
* cat x vars must be string/label encoded
* cat y must be in dummy var format

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#from sklearn.preprocessing import LabelEncoder #not needed if the data are preprocessed in spark and already in indexed form


#create a subclass of Datase
class pdDataset(Dataset):
	
	"""
	data: pandas df (e.g. from read.csv) from which we will use a subset of cols
	
	enc_cat_cols = list of strings:categorical column must already be encoded (tring encoded from spark, or label encoded in sklearn)
	output_cols = list of output and assumed to be numerical 
	float_cols = numerical variables
	
	"""	
	def __init__(self, data, enc_cat_cols = None, float_cols= None, output_cols = None):
			
		self.n = data.shape[0] #num rows
		self.y = data[output_cols].astype(np.float32).values #y vals
		self.float_x = data[float_cols].astype(np.float32).values #float x vals
		self.cat_x = data[enc_cat_cols].astype(np.int64).values #cat x, will be embedded later				
				
	def __len__(self):
		return self.n
	
	def __getitem__(self,idx):
		return [self.y[idx], self.float_x[idx], self.cat_x[idx]]


#define a feed forward neural net that uses embedding for categorical variables
class customNN(nn.Module):
	"""
	for each cat variable, emb_dim is a 'list' of 'tuples'. Where each tuple has 2 members, 
	the first is the number of unique categories and the size of the embedding vector space. 
	
	num_float: # of numerical x vars, output_size is num of output of target vars
	lin_layer_sizes: list of ints depicting number of units in each linear layer (size of list # layers) 
	
	lin_dropouts for each layer: a list of floats
	"""
	def __init__(self, emb_dims, num_float_x, lin_layer_sizes, output_size, emb_dropout, lin_layer_dropouts):
		super(customNN, self).__init__()
		
		#embedding layers
		self.emb_layers = nn.ModuleList( [nn.Embedding(x,y) for x,y in emb_dims])
		num_emb = sum([y for x,y in emb_dims]) #total dim over all embeddings		
		self.num_emb = num_emb
		
		self.num_float_x = num_float_x #total dim of all float vars
		
		##linear layers (post embedding)
		#1st layer from input (embed + all floats) to (layer_size[0]) hidden units 
		first_lin_layer = nn.Linear(self.num_emb + self.num_float_x, lin_layer_sizes[0])
		
		#list of layers
		self.lin_layers = nn.ModuleList([first_lin_layer] + 
				[nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i+1]) for i in range(len(lin_layer_sizes)-1)])	
		
		#initialize each layer in the list
		for lin_layer in self.lin_layers:
			nn.init.normal_(lin_layer.weight.data)
		
		#output layer
		self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
		nn.init.normal_(self.output_layer.weight.data)
				
		## Batch Norm Layers
		self.first_bn_layer = nn.BatchNorm1d(self.num_float_x)
		self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])
		#
		## Dropout Layers
		self.emb_dropout_layer = nn.Dropout(emb_dropout)
		self.dropout_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])
	
	
	def forward(self, float_x_data, cat_x_data):
		if self.num_emb != 0:
			x = [emb_layer(cat_x_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]
			x = torch.cat(x, 1) #concat dimension '1'. torch cat requires input as tuples
			x = self.emb_dropout_layer(x)
		
		if self.num_float_x != 0:
			normalized_cont_data = self.first_bn_layer(float_x_data)
		
			if self.num_emb != 0:
				x = torch.cat([x, normalized_cont_data], 1) #concat dimension 1, input as list or tuples
			else:
				x = normalized_cont_data
		
		#order of BN and dropout switched: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout-in-tensorflow	
		for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.dropout_layers, self.bn_layers):      
			x = F.relu(lin_layer(x))
			x = dropout_layer(x) #dropout before BN
			x = bn_layer(x)
	
		x = self.output_layer(x)
	
		return x


#numerical vars for neural network
float_x_vars = ["loan_amnt", "int_rate","annual_inc", "dti", "revol_util", "installment", "inst2inc"]
##StringEncoding of categorical variables: these will be inputs to embedding layer
cat_x_vars = ["term","grade","home_ownership", "pred_KM","emp_length"]

#float y vars: time vars are both original as well as counterfactuals from prepDataforNN.py 
y_vars = ['probDef','timeDef', 'probER', 'timeER']


varList = float_x_vars + cat_x_vars + y_vars
   
data = pd.read_csv("data/pdDataNN.csv",usecols= varList)#, nrows=640)

randList = np.random.rand(len(data))
randListIdx = randList < 0.8

train = data[randListIdx]
test = data[~randListIdx]

#use string/factor encoding to transform the dataset, if not already transformed

# str_encoder = {} #empty dict
# for cat_col in cat_x_vars:
# 	str_encoder[cat_col] = LabelEncoder()
# 	data[cat_col] = str_encoder[cat_col].fit_transform(data[cat_col])

#use something similar when y is categorical

dataset_train = pdDataset(data=train, enc_cat_cols=cat_x_vars, float_cols = float_x_vars, output_cols=y_vars)
dataset_test = pdDataset(data=test, enc_cat_cols=cat_x_vars, float_cols = float_x_vars, output_cols=y_vars)

batchsize = 64
dataloader_train = DataLoader(dataset_train, batchsize,shuffle=True, num_workers=1)
dataloader_test = DataLoader(dataset_test, batch_size = len(test),shuffle=False, num_workers=1)


#list of uniq levels for each cat variable 
cat_var_dims = [int(data[col].nunique()) for col in cat_x_vars]

#use this to create a list of 2-tuples for each cat var and dimensions for its embeddings
emb_dims = [(x, min(10, (x+1)//2)) for x in cat_var_dims] #max vector space 10

model = customNN(emb_dims=emb_dims, num_float_x=len(float_x_vars),
	 lin_layer_sizes = [50,50, 50], output_size=len(y_vars),
	 	emb_dropout = 0.01, lin_layer_dropouts = [0.01,0.01])

#Training
num_epochs = 10;
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(num_epochs):
	print('epoch %s .. ' % str(epoch))
	for y, float_x, cat_x in dataloader_train:
		
		#forward
		preds = model(float_x, cat_x)
		loss = criterion(preds,y)
		
		#backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

#preds:

#testing
for y, float_x, cat_x in dataloader_test:
	preds_test = model(float_x, cat_x)
	y_test = y

#convert to pandas dataframe
colList = ['probDefPred','timeDefPred', 'probERPred', 'timeERPred']+['probDef','timeDef', 'probER', 'timeER']

pdf = pd.DataFrame(torch.cat([preds_test,y_test],1).detach().numpy(), columns=colList)

#save to csv
pdf.to_csv('data/resultNNCounterFact.csv', index=False)

