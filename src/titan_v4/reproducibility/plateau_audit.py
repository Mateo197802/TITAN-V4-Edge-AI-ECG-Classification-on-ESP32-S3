"""Full-support, non-selective error accounting; no learned classifier or label edits."""
import numpy as np


def error_decomposition(y, prediction):
    if y.ndim!=1 or prediction.shape!=y.shape or not len(y) or not np.isin(y,range(9)).all() or not np.isin(prediction,range(9)).all():
        raise ValueError('Invalid full-support reference/prediction vector')
    cm=np.zeros((9,9),np.int64);np.add.at(cm,(y,prediction),1)
    errors=cm.sum(1)-cm.diagonal()
    confusions=[{'reference':i,'prediction':j,'records':int(cm[i,j])} for i in range(9) for j in range(9) if i!=j and cm[i,j]]
    return {'records':len(y),'errors':int(errors.sum()),'class_errors':errors.tolist(),
            'top_confusions':sorted(confusions,key=lambda r:(-r['records'],r['reference'],r['prediction']))}


def shared_errors(y, predictions):
    if predictions.ndim!=2 or predictions.shape[1]!=len(y) or not len(predictions):
        raise ValueError('Aligned model predictions required')
    for prediction in predictions:error_decomposition(y,prediction)
    correct=predictions==y[None,:]; pairs=[]
    for i in range(len(correct)):
        for j in range(i+1,len(correct)):
            pairs.append({'first':i,'second':j,'first_only_correct':int((correct[i]&~correct[j]).sum()),
                          'second_only_correct':int((correct[j]&~correct[i]).sum()),'both_wrong':int((~correct[i]&~correct[j]).sum())})
    return {'records':len(y),'all_models_wrong':int((~correct.any(0)).sum()),
            'any_model_correct':int(correct.any(0).sum()),'oracle_accuracy':float(correct.any(0).mean()),
            'deployable_classifier':False,'pairwise':pairs}
