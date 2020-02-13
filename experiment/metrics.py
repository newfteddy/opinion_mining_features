def map_clusters(y_true, y_pred):
    m = {}
    clusters = set(y_true)
    for c1 in clusters:
        cnt1 = 0
        for c2 in set(y_pred): 
            
            cnt = 0
            for (x,y) in zip(y_true,y_pred):
                if (x==c1) & (y==c2):
                    cnt+=1
            if cnt>cnt1:
                cnt1 = cnt
                res = c2
        m[c1] = res
    return m

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
def precision_recall(y_true,y_pred):
    m = map_clusters(y_pred,y_true)
    if len(set(m.values()))<len(set(y_true)):
        return 0,0
    y_true = np.array([m[x] for x in y_true])
    precision = metrics.precision_score(y_true,y_pred,average='weighted')
    recall = metrics.recall_score(y_true,y_pred,average='weighted')
    return precision,recall

def pairwise(y_pred,y_true):
    y_true_pairs = []
    for i in range(len(y_true)):
        for j in range(len(y_true)):
            if i == j:
                continue
            if y_true[i] == y_true[j]:
                y_true_pairs.append(sorted([i,j]))

    
    cnt = 0
    for pair in y_true_pairs:
        if y_pred[pair[0]] == y_pred[pair[1]]:
            cnt +=1
    size = len(y_true_pairs)
    return cnt/size

def pairwise2(y_pred,y_true):
    y_false_pairs = []
    for i in range(len(y_true)):
        for j in range(len(y_true)):
            if i == j:
                continue
            if y_true[i] != y_true[j]:
                y_false_pairs.append(sorted([i,j]))

    
    cnt = 0
    for pair in y_false_pairs:
        if y_pred[pair[0]] != y_pred[pair[1]]:
            cnt +=1
    size = len(y_false_pairs)
    return cnt/size