import pandas as pd
from sklearn.model_selection import train_test_split
from stc import *
from sklearn.metrics import mutual_info_score
from data import *
import sys
import time
from dim_reduction import *

# try except used so that we can set variables for easier testing in IDE
try:
    path_option = sys.argv[1]  # spam, amazon, or survival
except IndexError:
    path_option = "survival"
try:
    T = int(sys.argv[2])  # number of iterations
except IndexError:
    T = 10
try:
    LAMBDA = float(sys.argv[3])  # hyperparameter
except IndexError:
    LAMBDA = 1
try:
    NUM_FEATURES = int(sys.argv[4])  # length of vocabulary vectors
except IndexError:
    NUM_FEATURES = 1000
try:
    AUXILIARY_DATA_SIZE = int(sys.argv[5])  # number of examples to use from auxiliary dataset
except IndexError:
    AUXILIARY_DATA_SIZE = 1000
try:
    TARGET_DATA_SIZE = int(sys.argv[6])  # number of examples to use from target dataset
except IndexError:
    TARGET_DATA_SIZE = 100
try:
    dim_reduction_method = str(sys.argv[7])  # see all options README
except IndexError:
    dim_reduction_method = None

start_time = time.time()

if path_option == "spam":
    SHARED_COLS = [611]  # see paper for explanation regarding choice of shared feature space

    # list of lists, and list of 0s and 1s
    auxiliary_data, Y_labels = collect_spam_a_data("data\\task_a_labeled_train.tf",
                                                   num_features=NUM_FEATURES)
    # list of lists, and list of 0s and 1s
    target_data, X_labels = collect_spam_a_data("data\\task_a_u00_eval_lab.tf",
                                                num_features=NUM_FEATURES)

elif path_option == "amazon":
    SHARED_COLS = [0]  # see paper for for explanation regarding choice of shared feature space

    temp = collect_amazon_data("data\\Toys_and_Games_5.json",
                               "data\\Patio_Lawn_and_Garden_5_1.json",
                               num_features=NUM_FEATURES,
                               max_num_reviews=AUXILIARY_DATA_SIZE)

    auxiliary_data = temp[0]  # list of lists
    Y_labels = temp[1]  # list of 0s and 1s
    target_data = temp[2]  # list of lists
    X_labels = temp[3]  # list of 0s and 1s

elif path_option == "survival":
    # loading and cleaning Game of Thrones data
    target_data = pd.read_csv("440data\\game_of_thrones\\game_of_thrones.csv")
    target_data = target_data.loc[(target_data['title'].notnull())
                                  & (target_data['house'].notnull())
                                  & (target_data['isPopular'] == 1)
                                  & (target_data['dateOfBirth'].notnull())].drop(
        ["S.No", "actual", "pred", "alive", "plod", "DateoFdeath",
         "popularity", "title", "mother", "father", "heir",
         "isAliveMother", "isAliveFather", "isAliveHeir", "spouse",
         "isNoble", "boolDeadRelations"], axis=1)
    X_labels = list(target_data['isAlive'])
    X_names = list(target_data['name'])
    updated_X_labels = pd.read_csv("440data\\game_of_thrones\\game_of_thrones_updated.csv")
    updated_X_labels = updated_X_labels.drop(["Name", "isAlive"], axis=1)
    updated_X_labels = list(updated_X_labels['aliveCheck'])

    X_labels = updated_X_labels
    target_data = target_data.drop(['isAlive', 'name'], axis=1)
    # dateOfBirth, age, and numDeadRelations - round to nearest 10
    target_data = round_rows(target_data, ['dateOfBirth', 'age', 'numDeadRelations'])

    # loading and cleaning Titanic data
    # missing data for Age and Embarked - treat as additional category
    auxiliary_data = pd.read_csv("440data\\titanic\\titanic.csv")
    Y_labels = list(auxiliary_data['Survived'])
    # Fare and Age - round to nearest 10
    auxiliary_data = round_rows(auxiliary_data, ['Fare', 'Age'])
    # encode 'Sex' - male to 1, female to 0
    auxiliary_data['Sex'] = np.where(auxiliary_data['Sex'] == 'female', 0, 1)
    auxiliary_data['SibSp'] = np.where(auxiliary_data['SibSp'] > 0, 1, 0)
    auxiliary_data['Parch'] = np.where(auxiliary_data['Parch'] > 0, 1, 0)
    # Cabin has mostly null values
    auxiliary_data = auxiliary_data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Survived'], axis=1)

else:
    raise NotADirectoryError("Please enter a valid dataset.")

print("Loaded data")

if path_option == "survival":
    # discrete random variable corresponding to the common feature space of both X and Y
    Z_x = encode_shared_features(target_data[['male', 'age']])[:TARGET_DATA_SIZE]
    Z_y = encode_shared_features(auxiliary_data[['Sex', 'Age']])[:AUXILIARY_DATA_SIZE]
    Z = Z_x + Z_y
    Z_x = pd.Series(Z_x)
    Z_y = pd.Series(Z_y)
    Z = pd.Series(Z)
else:
    # discrete random variable corresponding to the common feature space of both X and Y
    Z_x = discretize_shared_features(target_data[:TARGET_DATA_SIZE], SHARED_COLS)[0]
    Z_y = discretize_shared_features(auxiliary_data[:AUXILIARY_DATA_SIZE], SHARED_COLS)[0]
    Z = Z_x + Z_y
    Z_x = pd.Series(Z_x)
    Z_y = pd.Series(Z_y)
    Z = pd.Series(Z)

# before discretizing data, perform PCA on both the auxiliary and target datasets if desired
if dim_reduction_method is not None:
    if dim_reduction_method == "pca_double" or dim_reduction_method == "sparse_double"\
            or dim_reduction_method == "truncated_double" or dim_reduction_method == "kernel_double":
        if path_option == "spam" and dim_reduction_method == "kernel_double":  # fixing weird bug
            auxiliary_data = [arr.tolist() for arr in auxiliary_data]
            target_data = [arr.tolist() for arr in target_data]
        auxiliary_data, target_data = perform_pca_double(auxiliary_data, target_data, dim_reduction_method)
    elif dim_reduction_method == "LPP":
        auxiliary_data = LPP(auxiliary_data)
        target_data = LPP(target_data)
    elif dim_reduction_method == "LPP_double":
        auxiliary_data, target_data = LPP(auxiliary_data, dataset2=target_data)
    else:
        auxiliary_data = perform_pca(auxiliary_data, dim_reduction_method)
        target_data = perform_pca(target_data, dim_reduction_method)

if path_option != "survival":
    # see discretize_data method in stc
    auxiliary_data = discretize_data(auxiliary_data)[:AUXILIARY_DATA_SIZE]
    target_data = discretize_data(target_data)[:TARGET_DATA_SIZE]
    print("Discretized data")

    # discrete random variable corresponding to the target data
    X = pd.Series(encode_rows(target_data))  # see encode_rows method in stc
    # discrete random variable corresponding to the auxiliary data
    Y = pd.Series(encode_rows(auxiliary_data))
    print("Encoded X, Y, and Z")
else:
    # discrete random variable corresponding to the target data
    X = pd.Series(encode_shared_features(target_data)[:TARGET_DATA_SIZE])  # see encode_rows method in stc
    # discrete random variable corresponding to the auxiliary data
    Y = pd.Series(encode_shared_features(auxiliary_data)[:AUXILIARY_DATA_SIZE])
    print("Encoded X, Y, and Z")

# get lists of possible X, Y, Z values
X_values = X.unique()
Y_values = Y.unique()
Z_values = Z.unique()

# initialize clusters (randomly - half in cluster 0 and half in cluster 1)
Cx = train_test_split(X_values)
Cy = train_test_split(Y_values)
Cz = train_test_split(Z_values)
print("Initialized clusters")

# create arrays corresponding to cluster values of each random variable
# i.e. arrays of 0s and 1s, where 0 and 1 are clusters
Xt = create_It(X, Cx)
Yt = create_It(Y, Cy)
Z_xt = create_It(Z_x, Cz)
Z_yt = create_It(Z_y, Cz)

# Pr(X), etc.
p_x = create_p_i(X, X_values)
p_y = create_p_i(Y, Y_values)
p_zx = create_p_i(Z_x, Z_values)
p_zy = create_p_i(Z_y, Z_values)
print("Created probability distributions")

# initialize p(X,Z) and q(Y,Z)
p_xz = create_joint_pdf(X, Z_x)
q_yz = create_joint_pdf(Y, Z_y)

# initialize pt(X,Z) = p(xt,zt)(px/pxt)(pz/pzt)
pt_XZ = create_p_tilde(X, Cx, Xt, X_values, Z_x, Cz, Z_xt, Z_values)
# initialize qt(Y,Z) = q(yt,zt)(qy/qyt)(qz/qzt)
qt_YZ = create_p_tilde(Y, Cy, Yt, Y_values, Z_y, Cz, Z_yt, Z_values)

current_X_score = mutual_info_score(X, Z_x) - mutual_info_score(Xt, Z_xt)
current_Y_score = mutual_info_score(Y, Z_y) - mutual_info_score(Yt, Z_yt)
total_score = current_X_score + LAMBDA * current_Y_score  # the value being minimized

# print("Starting accuracy on target data:", calculate_accuracy(X, Cx, X_labels))
# print("Starting accuracy on auxiliary data:", calculate_accuracy(Y, Cy, Y_labels))

for _ in range(T):
    if total_score != 0:  # the lowest (most desired) score we can get is zero, so no need to continue
        print("New iteration...")
        # C_x(x) = argmin(xt in Xt) D(p(Z|x)||pt(Z|xt))
        new_Cx = [[], []]
        for x in X_values:
            argmin = find_argmin_jt(Z_x, p_zx, Cz, Z_xt, X, Xt, x)  # see stc.py
            new_Cx[argmin].append(x)  # add this value of x to cluster 0 or 1

        # C_y(y) = argmin(yt in Yt) D(q(Z|y)||qt(Z|yt))
        new_Cy = [[], []]
        for y in Y_values:
            argmin = find_argmin_jt(Z_y, p_zy, Cz, Z_yt, Y, Yt, y)  # see stc.py
            new_Cy[argmin].append(y)  # add this value of y to cluster 0 or 1

        # C_z(z) = argmin(zt in Zt) p(z)D(p(X|z)||pt(X|zt))
        #           + lambda q(z)D(q(Y|z)||qt(Y|zt))
        new_Cz = [[], []]
        for z in Z_values:
            # see stc.py
            argmin = find_shared_argmin(X, p_x, Cx, Xt, Y, p_y, Cy, Yt, Z_x, Z_y, Z_xt, Z_yt, p_zx, p_zy, z, LAMBDA)
            new_Cz[argmin].append(z)  # add this value of z to cluster 0 or 1

        # update clusters associated with each example
        new_Xt = create_It(X, new_Cx)
        new_Z_xt = create_It(Z_x, new_Cz)
        new_Yt = create_It(Y, new_Cy)
        new_Z_yt = create_It(Z_y, new_Cz)
        new_X_score = mutual_info_score(X, Z_x) - mutual_info_score(new_Xt, new_Z_xt)
        new_Y_score = mutual_info_score(Y, Z_y) - mutual_info_score(new_Yt, new_Z_yt)

        # update clusters and scores
        Cx = new_Cx
        Cy = new_Cy
        Cz = new_Cz
        Xt = new_Xt
        Z_xt = new_Z_xt
        Yt = new_Yt
        Z_yt = new_Z_yt
        current_X_score = new_X_score
        current_Y_score = new_Y_score

        # pt(X,Z) = p(xt,zt)(px/pxt)(pz/pzt)
        pt_XZ = create_p_tilde(X, Cx, Xt, X_values, Z_x, Cz, Z_xt, Z_values)
        # qt(Y,Z) = q(yt,zt)(qy/qyt)(qz/qzt)
        qt_YZ = create_p_tilde(Y, Cy, Yt, Y_values, Z_y, Cz, Z_yt, Z_values)

# print("Finished training after %s seconds" % (time.time() - start_time))
# print(Cx)
# print(Cy)
# print(Cz)

print("Accuracy on target data:", calculate_accuracy(X, Cx, X_labels))
print("Accuracy on auxiliary data:", calculate_accuracy(Y, Cy, Y_labels))
