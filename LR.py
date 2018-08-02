import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
from scipy.sparse.linalg import svds
import io

def remove_small(table,row):
    temp_max=row.max()
    row=row.toarray()
    row[row<temp_max*0.1]=0
    row=csr_matrix(row)
    return vstack([table,row])

def extend_to_intercept(x):
    row,col=x.shape
    one_matrix=np.ones((row,1))
    combine=hstack([one_matrix,x])
    return combine

def logistic_predictions(weights, inputs):
    z = inputs.dot(weights)
    z=z.toarray()
    return 1.0/(1.0 + np.exp(-z))

def logistic_loss_and_grad(weights, inputs, targets):

#derivative of the loss function
    targets=targets.toarray()
    z = inputs.dot(weights).toarray()
    exp_z = np.exp(z)
    temp = -exp_z/(1 + exp_z) + targets
    m_t=csr_matrix(temp).multiply(inputs)
    sum_data=m_t.sum(axis=0)
    dloss = (-1)*np.array(sum_data)
    #calculating the loss is optional
    loss = (-1)*np.sum(-np.log(1.0 + exp_z) + targets*z)
    return np.transpose(dloss)

def fit(training_data,Y,n_iter,eta):
    X= training_data
    N, D = X.shape
    weights = np.random.random(D)[:, np.newaxis]
    for epoch in range(n_iter):
        dloss= logistic_loss_and_grad(csr_matrix(weights), X, Y)
        weights = weights  - dloss * eta 
    return np.transpose(weights)

def my_predict(predict_table,all_labels):
    max_predict=list()
    l=predict_table.shape[0]
    for i in range(0,l):
        index_n=np.argmax(predict_table[i,:])
        max_predict.append(all_labels[index_n])
    return np.array(max_predict)

def score_performance(p,c):
    count=0
    l=p.shape[0]
    for i in range(0,l):
        if p[i]==c[i]:
            count=count+1
    return float (count)/float(l)

# Perform K fold validation, default is 10
def K_fold_validation(table, labels):
    K=10
    rows,columns=table.shape
    reconstruct_table, pre_labels , all_labels=Reconstruct_data(table,labels)
    test_length=float(rows/K)
    
    for i in range(K):
        start=int(i*test_length)
        finish=int((i+1)*test_length)
        test_data=reconstruct_table[start:finish,:]
        training_data=vstack([reconstruct_table[0:start,:],reconstruct_table[finish:rows,:]])
        correct_labels=np.array(labels[start:finish])
        training_labels=np.vstack((pre_labels[0:start,:],pre_labels[finish:rows,:]))
        Multi_class_LR(training_data,training_labels,test_data,correct_labels,all_labels,columns)

def Reconstruct_data(table, labels):
    # reconstruct table, reomove the small value in a row
    rows,columns=table.shape
    sparse_table=csr_matrix(table)
    temp_table=sparse_table[0,:]
    temp_max=temp_table.max()
    temp_table=temp_table.toarray()
    temp_table[temp_table<temp_max*0.1]=0
    reconstruct_table=csr_matrix(temp_table)
    for x in range(1,rows):
        reconstruct_table=remove_small(reconstruct_table,sparse_table[x,:])

    # find all the 30 labels
    all_labels=sorted((set(labels[1])))
    
    # reconstruct the labels, make them into a understandable form
    pre_labels = np.zeros(shape=(rows,30))
    i=0
    for word in all_labels:
        index=np.array(labels.index[labels[1].str.strip()==word])
        pre_labels[index,i]=1
        i=i+1
    pre_labels=csr_matrix(pre_labels)

    return reconstruct_table, pre_labels , all_labels

def Multi_class_LR(training_data,training_labels,test_data,correct_labels,all_labels,columns):

    weights_table=np.zeros(shape=(columns,30))
    for i in range(0,30):
        eta=0.5
        n_iter=1500
        weights_table[:,i]=fit(training_data,training_labels[:,i],n_iter,eta)
        print(i)
        predict_table=logistic_predictions(csr_matrix(weights_table),test_data)
        predict_labels=my_predict(predict_table,all_labels)
        accuracy=score_performance(predict_labels,correct_labels)
        print(accuracy)



if __name__ == '__main__':
    #loading data
    table = pd.read_csv('training_data.csv',delimiter=',',header=None,index_col=0).sort_index()
    labels = pd.read_csv('training_labels.csv',delimiter=',',header=None,index_col=0).sort_index()
    df_test = pd.read_csv('test_data.csv',delimiter=',',header=None,index_col=0)
    labels=labels.reset_index(drop=True)
    table=table.reset_index(drop=True)
    
    K_fold_validation(table, labels)