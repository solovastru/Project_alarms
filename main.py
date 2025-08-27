import numpy as np
import pandas as pd
from topsis_module import Topsis
from ahp_module import ahp_weights
from inference import InferBN
from pgmpy.models import DiscreteBayesianNetwork


oee_spc_importance = {("oee", "oee"): 1, ("oee", "spc"): 2,
           ("spc", "oee"): 1/2, ("spc", "spc"): 1}


model = DiscreteBayesianNetwork.load('bayesian_tu_project.bif', filetype='bif') 
#observed evidence
obs_alarms = {"Alarm1": "0", "Alarm2": "1", "Alarm3": "0","Alarm4": "0","Alarm5": "1"}

#get the inference class
inference = InferBN(model)

#do inference based on the observed alarms
my_matrix = inference.infer(observed_alarms=obs_alarms)


#compute the weight for the OEE and SPARE COST criteria
oee_spc_weights = list(map(float, ahp_weights(oee_spc_importance).values()))

#transform into array the df with the probabilities for the highest OEE and the lowest SPARE_COST
evaluation_matrix = my_matrix[["High_OEE", "Low_SPARE_COST"]].to_numpy()


criterion =  ['max', 'max']

topsis = Topsis(my_matrix, oee_spc_weights, criterion)
ranked_df = topsis.evaluate()

print(ranked_df)
