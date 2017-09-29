import numpy as np
import random
import matplotlib.pyplot as plt
import sys

def rotate_files(files):
    files.append(files.pop(0))
    return files  

def create_data(filepath):
    y = []
    x = []
    with open(filepath,'r') as file:
            for line in file:
                y.append(int(line[:2]))
                obs = [0] * 68
                obs.append(1)
                for l in line[2:].split():
                    for i,w in enumerate(l):
                        if(w == ":"):
                            obs[int(l[:i])-1] = float(l[i+1:])
                x.append(obs)           
    return x,np.array(y)

def perceptron(rows, Y, w, updates,margin):
    for i, x in enumerate(rows):
        if Y[i] * np.sum(w * np.array(x)) < margin :
            n = (margin - np.sum(Y[i] * w * np.array(x)))/np.sum((np.array(x))**2)
            w += (n*Y[i])*np.array(x)
            updates +=1
    return updates, w

def predict(weight,test_data):
    result = []
    for x in test_data:
        if np.sum(weight * np.array(x)) < 0:
            result.append(-1)
        else:
            result.append(1)            
    return result  

def cross_validation(cv_sets, weight,margin):
    for i in range(len(cv_sets)):
        updates = 0
        w = weight
        cross_val_acc = []
        cv_y_test = []
        files = rotate_files(cv_sets)
        cv_train_set = files[0:4]
        cv_test_set = files[4]
        cv_X = list()
        cv_Y =list()
        for file in cv_train_set:
            cv_x, cv_y = create_data(file)
            cv_X.extend(cv_x)
            cv_Y.extend(cv_y)
        for i in range(0, 10):
            random.seed(i*10)
            random.shuffle(cv_X)
            random.seed(i*10)
            random.shuffle(cv_Y)
            updates, w = perceptron(cv_X, cv_Y, w, updates,margin)
        cv_x_test, cv_y_test_act = create_data(cv_test_set) 
        cv_y_test = predict(w, cv_x_test)
        cross_val_acc.append(np.mean(cv_y_test == cv_y_test_act)) 
        cross_val_result = np.mean(np.array(cross_val_acc))
    return cross_val_result 


def best_learn_rate(sets, w, margins):
    best_hyper = {}
    for margin in margins:
        best_hyper[margin] = cross_validation(sets, w, margin)
    print("Best Hyper-parameter Accuracy: " + str(max(best_hyper.values())))
    best_rate = [k for k,v in best_hyper.items() if v == max(best_hyper.values())][0]
    print("Best Hyper-parameter (Margin): " + str(best_rate))
    return best_rate

def perceptron_main(train_set, dev_set, test_set,sets, margins):
    count_updates = 0
    initial_w = np.array([0.001] * 69)
    margin = best_learn_rate(sets,initial_w, margins)
    X_dev, Y_dev = create_data(dev_set)
    X_test, Y_test = create_data(test_set)
    cross_val_accuracy_dev =[]
    pred_y_test = []
    best_wt = []
    max_acc = 0
    for i in range(0, 20):
        accuracy = 0
        X = []
        Y = []
        pred_dev_y = []
        X, Y = create_data(train_set)
        random.seed(i*123)
        random.shuffle(X)
        random.seed(i*123)
        random.shuffle(Y)
        count_updates, w = perceptron(X, Y,initial_w, count_updates,margin) 
        pred_dev_y = predict(w, X_dev)
        accuracy = np.mean(pred_dev_y == Y_dev)
        if accuracy > max_acc:
            max_acc = accuracy
            best_weight = w
        cross_val_accuracy_dev.append(accuracy) 
    pred_y_test =predict(best_weight, X_test)
    test_accuracy = np.mean(pred_y_test == Y_test) 
    print("Number of updates on training :" + str(count_updates))
    return test_accuracy, cross_val_accuracy_dev

cv0 = sys.argv[1]
cv1 = sys.argv[2]
cv2 = sys.argv[3]
cv3 = sys.argv[4]
cv4 = sys.argv[5]
train_set = sys.argv[6]
_dev_set = sys.argv[7]
_test_set = sys.argv[8]


if __name__ == "__main__":
    
    cv_sets = [cv0,cv1,cv2,cv3,cv4,cv4]
    
    tr_set = train_set
    dev_set = _dev_set
    test_set = _test_set
    
    print("running for aggressive perceptron")
    acc_test, epoc_acc = perceptron_main(tr_set, dev_set, test_set,cv_sets, [1, 0.1, 0.01])
    print("Development set Accuracy",epoc_acc)
    print("Test set Accuracy",acc_test)
    #plt.plot(range(1, 21), epoc_acc)
    #plt.show()
