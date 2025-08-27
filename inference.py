
from pgmpy.inference import  VariableElimination, CausalInference
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
#load the trained model



class InferBN:
    def __init__(self, model_file):
        #load the model
        self.model = model_file
        self.inference = VariableElimination(self.model)
    
    def compute_actions_probs(self, observed_alarms: dict):
        # compute the probabilities for the maintenance actions given the alarms states
        maint_actions_probs = self.inference.query(["Maintenance_action"], evidence=observed_alarms)
        return dict(zip(
            maint_actions_probs.state_names['Maintenance_action'],
            map(float, maint_actions_probs.values)
        ))
    
    def compute_joint_probabilities(self, maint_actions_dict: dict):
        #for each high OEE and low SPARE_COST compute joint probabilities, because they depend on the probabilities from the actions
        rows = []
        for action, prior_prob in maint_actions_dict.items():
            evidence = {"Maintenance_action": action}

            oee_probs = self.inference.query(variables=['OEE'], evidence=evidence)
            spc_probs = self.inference.query(variables=['SPARE_COST'], evidence=evidence)

            oee_dict = dict(zip(oee_probs.state_names['OEE'], oee_probs.values))
            spc_dict = dict(zip(spc_probs.state_names['SPARE_COST'], spc_probs.values))

            # Multiply by prior of maintenance action
            prob_high_oee = prior_prob * oee_dict.get("High", 0)
            prob_low_spc = prior_prob * spc_dict.get("Low", 0)

            rows.append({
                "Maintenance_action": action,
                "High_OEE": prob_high_oee,
                "Low_SPARE_COST": prob_low_spc
            })
        
        df = pd.DataFrame(rows)
        return df

    def infer(self, observed_alarms: dict):
        dict_actions = self.compute_actions_probs(observed_alarms)
        result = self.compute_joint_probabilities(dict_actions)
        return result