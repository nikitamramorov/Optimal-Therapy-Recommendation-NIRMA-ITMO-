from .genetic import get_treatment

meds = ['perindopril', 'metformin', 'empagliflozin', 'linagliptin',
        'amlodipine', 'simvastatin', 'lisinopril dihydrate', 'insulin aspart',
        'insulin glargine', 'insulin lispro', 'losartan', 'gliclazide',
        'linagliptin / metformin', 'hydrochlorothiazide', 'telmisartan',
        'insulin isophane', 'glimepiride', 'labetalol']

def predict(X):
    X = dict(X)
    X.update(dict(zip(meds, [0] * 18)))
    return dict(zip(meds, get_treatment(X)))