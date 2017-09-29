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

def margin_perceptron(rows, Y, learn_rate, w, updates,margin):
    for i, x in enumerate(rows):
        if Y[i] * np.sum(w * np.array(x)) < margin :
            w += (learn_rate*Y[i])*np.array(x)
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

def cross_validation(cv_sets, learn_rate, weight,margin):
    for i in range(len(cv_sets)):
        w = weight
        cross_val_acc =[]
        cv_y_test = []
        updates = 0
        time_step = 0
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
            updates, w = margin_perceptron(cv_X, cv_Y, learn_rate, w, updates,margin)
        cv_x_test, cv_y_test_act = create_data(cv_test_set) 
        cv_y_test = predict(w, cv_x_test)
        cross_val_acc.append(np.mean(cv_y_test == cv_y_test_act)) 
        cross_val_result = np.mean(np.array(cross_val_acc))
    return cross_val_result 

def best_learn_rate(sets, rates, w,margin):
    best_rate = {}
    for rate in rates:
        best_rate[rate] ={} 
        for m in margin:
            best_rate[rate][m]=cross_validation(sets, rate, w,m)
    best=0
    best_margin = 0
    best_l_rate= 0
    for rate in rates:
        for m in margin:
            if best < best_rate.get(rate).get(m):
                best = best_rate.get(rate).get(m)
                best_l_rate = rate
                best_margin = m
    #best_l_rate = [k for k,v in best_rate.items() if v == max(best_rate.values())][0]
     
    print("Best Learning Rate: ",best_l_rate)
    print("Best Margin: ",best_margin)
    print("Best Cross validation Accuracy: " + str(best))
    return best_l_rate,best_margin

def perceptron_main(train_set, dev_set, test_set,sets, rates,margin):
    count_updates = 0
    initial_w = np.array([0.01] * 69)
    learn_rate,margin = best_learn_rate(sets, rates, initial_w,margin)
    X_dev, Y_dev = create_data(dev_set)
    X_test, Y_test = create_data(test_set)
    cross_val_accuracy_dev =[]
    pred_y_test = []
    best_wt = []
    max_acc = 0
    time_step = 0
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
        count_updates, w = margin_perceptron(X, Y, learn_rate, initial_w, count_updates,margin) 
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
    
    learn_rate = [1, 0.1, 0.01]
    margin = [1, 0.1, 0.01]
    print("running for margin Perceptron")
    acc_test, epoc_acc = perceptron_main(tr_set, dev_set, test_set,cv_sets,learn_rate,margin )
    print("Development set Accuracy",epoc_acc)
    print("Test set Accuracy",acc_test)
    #plt.plot(range(1, 21), epoc_acc)
    #plt.show()

