import bnlearn as bn
from joblib import load
import numpy as np
from simanneal import Annealer
import random

map_scaler = load('map_scaler.data')
hba1c_scaler = load('hba1c_scaler.data')
ldl_scaler = load('ldl_scaler.data')

map_cols = load('map_cols.data')
hba1c_cols = load('hba1c_cols.data')
ldl_cols = load('ldl_cols.data')
bn_cols = load('bn_cols.data')
med_cols = load('med_columns.data')

hc_bic_estimator = load('hc_bic_estimator.data')
hc_bic_cols = load('hc_bic_cols.data')

num_cols_hba1c = load('num_cols_hba1c.data')
num_cols_map = load('num_cols_map.data')
num_cols_ldl = load('num_cols_ldl.data')

map_model = load('map.model')
hba1c_model = load('hba1c.model')
ldl_model = load('ldl.model')

med_groups = {
    'insulins': ['insulin_ultrashort', 'insulin_short', 'insulin_medium', 'insulin_long', 'insulin_comb'],
    'all_gipoglemic_drugs': ['vildagliptin', 'glibenclamide', 'gliclazide', 'glimepiride', 'dapagliflozin', 'metformin', 'sitagliptin'],
    'all_statins': ['atorvastatin', 'rosuvastatin', 'simvastatin'],
    'diuretic': ['acetazolamide', 'hydrochlorothiazide', 'indapamide', 'urea', 'spironolactone', 'torasemide', 'furosemide', 'eplerenone'],
    'beta_block': ['bisoprolol', 'metoprolol', 'nebivolol', 'sotalol']
}

map_min = 67.96296296296295
map_max = 227.5
def scale_map(x):
    return (x - map_min) / (map_max - map_min)

ldl_min = 0.22
ldl_max = 6.68
def scale_ldl(x):
    return (x - ldl_min) / (ldl_max - ldl_min)

hba1c_min = 5.1631637
hba1c_max = 20.98978585
def scale_hba1c(x):
    return (x - hba1c_min) / (hba1c_max - hba1c_min)

def get_best_comb(q):
    combs = q.assignment(list(range(32)))
    max_p = -1
    ind = -1
    for t in range(len(combs)):
        p = q.get_value(**dict(combs[t]))
        if p > max_p:
            max_p = p
            ind = t
    return ind

def select_groups(X, estimator):
    X_d = discretize_row(X.to_dict())
    q = bn.inference.fit(estimator,
                         variables=['insulins', 'all_gipoglemic_drugs', 'all_statins', 'diuretic', 'beta_block'],
                         evidence=X_d,
                         verbose=0)
    index = get_best_comb(q)
    return q.assignment([index])[0]

def discretize_row(data):
    discretized = dict()
    if 'total_bilirubin_mean' in data:
        if data['total_bilirubin_mean'] < 3.4:
            discretized['total_bilirubin_mean'] = 'low'
        elif 3.4 <= data['total_bilirubin_mean'] <= 20.5:
            discretized['total_bilirubin_mean'] = 'normal'
        else:
            discretized['total_bilirubin_mean'] = 'high'
    if 'triglycerides_mean' in data:
        if data['triglycerides_mean'] < 1.8:
            discretized['triglycerides_mean'] = 'normal'
        elif 1.8 <= data['triglycerides_mean'] < 2.3:
            discretized['triglycerides_mean'] = 'borderline high'
        elif 2.3 <= data['triglycerides_mean'] < 5.7:
            discretized['triglycerides_mean'] = 'high'    
        else:
            discretized['triglycerides_mean'] = 'very high'
    if 'creatinine_mean' in data:   
        if data['creatinine_mean'] < 62:
            discretized['creatinine_mean'] = 'low'
        elif 62 <= data['creatinine_mean'] <= 106:
            discretized['creatinine_mean'] = 'normal'
        else:
            discretized['creatinine_mean'] = 'high'
    if 'potassium_mean' in data:    
        if data['potassium_mean'] < 3.5:
            discretized['potassium_mean'] = 'low'
        elif 3.5 <= data['potassium_mean'] <= 5.5:
            discretized['potassium_mean'] = 'normal'
        else:
            discretized['potassium_mean'] = 'high'
    if 'AST_mean' in data:
        if data['AST_mean'] <= 31:
            discretized['AST_mean'] = 'normal'
        else:
            discretized['AST_mean'] = 'high'
    if 'ALT_mean' in data:
        if data['ALT_mean'] <= 32:
            discretized['ALT_mean'] = 'normal'
        else:
            discretized['ALT_mean'] = 'high'
    if 'sodium_mean' in data:    
        if data['sodium_mean'] < 130:
            discretized['sodium_mean'] = 'low'
        elif 130 <= data['sodium_mean'] <= 156:
            discretized['sodium_mean'] = 'normal'
        else:
            discretized['sodium_mean'] = 'high'
    if 'total_protein_mean' in data:      
        if data['total_protein_mean'] < 65:
            discretized['total_protein_mean'] = 'low'
        elif 65 <= data['total_protein_mean'] <= 85:
            discretized['total_protein_mean'] = 'normal'
        else:
            discretized['total_protein_mean'] = 'high'
    if 'HDL_mean' in data:     
        if data['HDL_mean'] > 1.02:
            discretized['HDL_mean'] = 'target level'
        else:
            discretized['HDL_mean'] = 'nontarget level'
    if 'hba1c_mean' in data:      
        if data['hba1c_mean'] > 6:
            discretized['hba1c_mean'] = 'Normal'
        elif 6.0 <= data['hba1c_mean'] < 7.0:
            discretized['hba1c_mean'] = 'Controlled'
        else:
            discretized['hba1c_mean'] = 'Uncontrolled'
    if 'LDL_mean' in data:
        if data['LDL_mean'] < 2.6:
            discretized['LDL_mean'] = 'target level'
        else:
            discretized['LDL_mean'] = 'nontarget level'
    if 'cholesterol_mean' in data:   
        if data['cholesterol_mean'] < 4.5:
            discretized['cholesterol_mean'] = 'normal'
        else:
            discretized['cholesterol_mean'] = 'high'
    if 'AC_mean' in data:       
        if data['AC_mean'] < 3.5:
            discretized['AC_mean'] = 'normal'
        else:
            discretized['AC_mean'] = 'high'
    if 'hemoglobin_mean' in data:     
        if data['hemoglobin_mean'] < 130:
            discretized['hemoglobin_mean'] = 'low'
        elif 130 <= data['hemoglobin_mean'] <= 160:
            discretized['hemoglobin_mean'] = 'normal'
        else:
            discretized['hemoglobin_mean'] = 'high'
    if 'hematocrit_mean' in data:      
        if data['hematocrit_mean'] < 40:
            discretized['hematocrit_mean'] = 'low'
        elif 40 <= data['hematocrit_mean'] <= 48:
            discretized['hematocrit_mean'] = 'normal'
        else:
            discretized['hematocrit_mean'] = 'high'
    if 'leukocytes_mean' in data:    
        if data['leukocytes_mean'] < 4.0:
            discretized['leukocytes_mean'] = 'low'
        elif 4.0 <= data['leukocytes_mean'] <= 9.0:
            discretized['leukocytes_mean'] = 'normal'
        else:
            discretized['leukocytes_mean'] = 'high'
    if 'CAD_mean' in data:    
        if data['CAD_mean'] < 90.0:
            discretized['CAD_mean'] = 'low'
        elif 90.0 <= data['CAD_mean'] <= 120.0:
            discretized['CAD_mean'] = 'normal'
        elif 120.0 < data['CAD_mean'] < 140.0:
            discretized['CAD_mean'] = 'prehypertension'
        else:
            discretized['CAD_mean'] = 'high'
    if 'DAD_mean' in data:     
        if data['DAD_mean'] < 60.0:
            discretized['DAD_mean'] = 'low'
        elif 60.0 <= data['DAD_mean'] <= 80.0:
            discretized['DAD_mean'] = 'normal'
        elif 80.0 < data['DAD_mean'] < 90.0:
            discretized['DAD_mean'] = 'prehypertension'
        else:
            discretized['DAD_mean'] = 'high'
    if 'bmi_mean' in data:     
        if data['bmi_mean'] < 16:
            discretized['bmi_mean'] = 'very low'
        elif 16.0 <= data['bmi_mean'] < 18.5:
            discretized['bmi_mean'] = 'underweight'
        elif 18.5 <= data['bmi_mean'] < 25.0:
            discretized['bmi_mean'] = 'normal'
        elif 25.0 <= data['bmi_mean'] < 30.0:
            discretized['bmi_mean'] = 'overweight'
        elif 30.0 <= data['bmi_mean'] < 35.0:
            discretized['bmi_mean'] = '1st degree obesity'
        elif 35.0 <= data['bmi_mean'] < 40.0:
            discretized['bmi_mean'] = '2nd degree obesity'
        else:
            discretized['bmi_mean'] = '3rd degree obesity'
    for c in ['arterial_hypertension', 'essential_hypertension', 'CHF', 'COPD', 'atherosclerosis', 'anemia', 'APC', 'CKD',
             'harmful_lifestyle', 'AF', 'IGT', 'metabolic_syndrome', 'obesity', 'stenocardia', 'sleep_disorders', 'DLM',
             'hyperglycemia', 'diabetic_osteoarthropathy', 'diabetic_ulcer', 'thyrotoxicosis', 'hypocorticism',
             'acromegaly', 'ischemic_cardiomyopathy', 'myocardial_infarction', 'CHD', 'ACS', 'diabetic_retinopathy',
             'diabetic_angiopathy', 'diabetic_nephropathy', 'neuropathy', 'cushing_syndrome', 'stroke']:
        if c in data:
            discretized[c] = 'Yes' if data[c] == 1 else 'No'
    return discretized

class BnAnnealer(Annealer):
    def __init__(self, state, row, bn_estimator, bn_columns):
        super(BnAnnealer, self).__init__(state)
        self.allowed_drugs = []
        groups = select_groups(row[bn_columns], bn_estimator)
        for group, answer in groups:
            if answer == 'Yes':
                self.allowed_drugs.extend(med_groups[group])
            else:
                for not_allowed_group in med_groups[group]:
                    self.state[not_allowed_group] = 0
        self.row_hba1c = row.to_frame().T.copy()
        self.row_hba1c[num_cols_hba1c] = hba1c_scaler.transform(self.row_hba1c[num_cols_hba1c])
        self.row_ldl = row.to_frame().T.copy()
        self.row_ldl[num_cols_ldl] = ldl_scaler.transform(self.row_ldl[num_cols_ldl])
        self.row_map = row.to_frame().T.copy()
        self.row_map[num_cols_map] = map_scaler.transform(self.row_map[num_cols_map])
                    
    def move(self):
        key = random.choice(self.allowed_drugs)
        if self.state[key] == 0:
            self.state[key] = 1
        else:
            self.state[key] == 0
    
    def energy(self):
        for drug in self.state:
            self.row_hba1c[drug] = self.state[drug]
            self.row_ldl[drug] = self.state[drug]
            self.row_map[drug] = self.state[drug]
        hba1c = hba1c_model.predict(self.row_hba1c[hba1c_cols])[0]
        ldl = ldl_model.predict(self.row_ldl[ldl_cols])[0]
        map_ = map_model.predict(self.row_map[map_cols])[0]
        return (np.abs(scale_map(map_) - scale_map(81.5)) * 0.5) + (np.abs(scale_hba1c(hba1c) - scale_hba1c(6.5)) * 0.2) + (np.abs(scale_ldl(ldl) - scale_ldl(2)) * 0.3)

def get_drugs(X):
    init_state = {}
    for drug in med_cols:
        init_state[drug] = random.randint(0,1)
    solver = BnAnnealer(init_state, X, hc_bic_estimator, hc_bic_cols)
    solver.steps = 1000
    itinerary, miles = solver.anneal()
    return itinerary
