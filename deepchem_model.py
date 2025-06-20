import deepchem as dc
import numpy as np
from sklearn.metrics import confusion_matrix

loader = dc.data.CSVLoader(
    tasks=['ESOL predicted log solubility in mols per litre'],
    smiles_field='smiles',
    featurizer=dc.feat.ConvMolFeaturizer()
)


dataset = loader.create_dataset('delaney-processed.csv')


splitters = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitters.train_valid_test_split(dataset)


model = dc.models.GraphConvModel(n_tasks=1, mode='regression')
model.fit(train_dataset, nb_epoch=50)


rmse_metric = dc.metrics.Metric(dc.metrics.mean_squared_error, mode='regression')
train_scores = model.evaluate(train_dataset, [rmse_metric])
valid_scores = model.evaluate(valid_dataset, [rmse_metric])
test_scores = model.evaluate(test_dataset, [rmse_metric])


print("\n=== Final Test RMSE ===")
print(f"Test RMSE: {test_scores['mean_squared_error']:.2f}")