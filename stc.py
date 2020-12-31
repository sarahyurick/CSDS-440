from math import log2
import random


def discretize_data(df):
    """
    Given: list of lists of numbers
    Output: list of lists of 0s and 1s
    """
    new_df = []
    for arr in df:
        nested_arr = []
        for value in arr:
            if value > 0:  # if number > 0, replace with 1
                nested_arr.append(1)
            else:  # if number <= 0, replace with 0
                nested_arr.append(0)
        new_df.append(nested_arr)
    return new_df


def round_rows(df, col_names):
    """
    Given: a dataset and the column names of real-valued features
    Output: the dataset with the real-valued features rounded to the closet 10
    """
    for col in col_names:
        new_col = []
        current_col = df[col]
        for value in current_col:
            try:
                new_value = round(value/10)*10
            except ValueError:
                new_value = 0
            new_col.append(new_value)
        df[col] = new_col
    return df


def encode_rows(df):
    """
    Given: list of lists
    Output: list of integers, where indices with the same integer represent matching lists from the list of lists
    """
    dup_free = set(tuple(row) for row in df)
    dup_free = list(dup_free)

    encodings = []
    for arr in df:
        for arr_index in range(len(dup_free)):
            if list(dup_free[arr_index]) == arr:
                encodings.append(arr_index)
                break
    return encodings


def discretize_shared_features(df, shared_cols):
    """
    Similar to discretize_data, except we are assuming a df of raw counts,
    so any values != 0 will also be > 0 and changed to 1,
    while all zeroes remain unchanged.
    """
    encoded = []
    for col in shared_cols:
        col_arr = []
        for val in df:
            if val[col] != 0:
                col_arr.append(1)
            else:
                col_arr.append(0)
        encoded.append(col_arr)
    return encoded


def encode_shared_features(df):
    """
    Prepares a dataframe to be used in encode_rows
    """
    list_of_lists = []
    for index, row in df.iterrows():
        list_of_lists.append(list(row))
    return encode_rows(list_of_lists)


def create_It(I, Ci):
    """
    Given: a list of values and the clusters of those values
    Output: a list of cluster assignments
    """
    It = []
    for i in I:
        # if Ci[0].contains(i):
        if i in Ci[0]:
            It.append(0)
        elif i in Ci[1]:
            It.append(1)
        else:
            raise RuntimeError("All values must be assigned to a cluster")

    return It


def create_p_i(I, I_values):
    """
    Given: a list of values, and all possible unique value in the list
    Output: a dict of probabilities associated with each value
    """
    p_i = dict()
    if len(I) == 0:
        return p_i

    for i in I_values:
        try:
            percentage = I.isin([i]).sum(axis=0) / len(I)
        except AttributeError:
            percentage = I.count(i) / len(I)
        entry = {i: percentage}
        p_i.update(entry)
    return p_i


def create_joint_pdf(I, J):
    """
    Given: 2 lists of values
    Output: a dict of dicts of probabilities associated with their joint probabilities
    """
    p_ij = dict()
    if len(I) != len(J):
        raise RuntimeError("The distributions of random variables must come from the same dataset")
    else:
        length = len(I)

    for i, j in zip(I, J):
        try:
            count = p_ij[i][j] * length
            count = count + 1
            percentage = count / length
            p_ij[i].update({j: percentage})
        except KeyError:
            try:
                p_ij[i].update({j: 1 / length})
            except KeyError:
                p_ij.update({i: dict()})
                p_ij[i].update({j: 1 / length})
    return p_ij


def find_associated_cluster(i, Ci):
    """
    Given: a value and clusters
    Output: the cluster (0 or 1) that the value is in
    """
    if i in Ci[0]:
        it = 0
    elif i in Ci[1]:
        it = 1
    else:
        raise RuntimeError("All values must be assigned to a cluster")
    return it


def safe_divide(a, b):
    """
    Just in case.
    """
    if b != 0:
        return a / b
    else:
        if a != 0:
            return 1000000
        else:
            return 0


def create_p_tilde(I, Ci, It, I_values, J, Cj, Jt, J_values):
    """
    See definition of p_tilde(x,z) in section 3.2 of STC paper, or Sarah's paper
    """
    p_itjt = create_joint_pdf(It, Jt)
    p_i = create_p_i(I, I_values)
    p_j = create_p_i(J, J_values)
    p_it = create_p_i(It, [0, 1])
    p_jt = create_p_i(Jt, [0, 1])

    p_tilde = dict()

    for i in I_values:
        pi = p_i[i]
        it = find_associated_cluster(i, Ci)
        pit = p_it[it]

        p_tilde.update({i: dict()})

        for j in J_values:
            pj = p_j[j]
            jt = find_associated_cluster(j, Cj)
            pjt = p_jt[jt]

            try:
                pitjt = p_itjt[it][jt]
            except KeyError:
                pitjt = 0

            pi_pit = safe_divide(pi, pit)
            pj_pjt = safe_divide(pj, pjt)
            pt = pitjt * pi_pit * pj_pjt
            p_tilde[i].update({j: pt})

    return p_tilde


def get_pt_I_given_jt(I, p_i, Ci, It, Jt, jt, p_it, p_jt):
    """
    See definition of p_tilde(Z|x_tilde) in section 3.2 of STC paper, or Sarah's paper
    """
    # To get pt(Z|xt) and qt(Z|yt)
    # Generalized to pt(I|jt)
    # pt(j,i) = p(j)pt(i|jt) = p(j) (p(jt,it)/p(jt)) (p(i)/p(it))

    pt_I_given_jt = dict()
    pjt = p_jt[jt]

    for i in I:
        pi = p_i[i]
        it = find_associated_cluster(i, Ci)
        pit = p_it[it]

        p_jtit = create_joint_pdf(Jt, It)
        try:
            pjtit = p_jtit[jt][it]
        except KeyError:
            pjtit = 0

        try:
            prob = (pjtit / pjt) * (pi / pit)
        except ZeroDivisionError:
            prob = 0
        pt_I_given_jt.update({i: prob})

    return pt_I_given_jt


def get_p_I_given_j(I, J, given_j):
    """
    Basic computation of conditional probability distribution, i.e. Pr(I|j)
    """
    new_I = []
    I_values = I.unique()

    for i, j in zip(I, J):
        if j == given_j:
            new_I.append(i)

    return create_p_i(new_I, I_values)


# calculate the kl divergence
def kl_divergence(p, q):
    """
    Given: 2 probability distributions
    Output: KL divergence (see definition in Sarah's paper)
    """
    p = list(p)
    q = list(q)
    total_sum = 0
    non_zero = False

    for i in range(len(p)):
        if p[i] != 0 and q[i] != 0:
            non_zero = True
            p_q = safe_divide(p[i], q[i])
            total_sum = total_sum + (p[i] * log2(p_q))

    if non_zero:
        return total_sum
    else:
        # raise ValueError("One of the input distributions was a list of zeroes")
        return 1000000


def find_argmin_jt(I, p_i, Ci, It, J, Jt, j):
    """
    See equations 14 and 15 in STC paper, or Sarah's paper.
    Outputs best cluster (0 or 1) for value j.
    """
    # for example, I = Z and J = X
    p_Ij = get_p_I_given_j(I, J, j)
    p_it = create_p_i(It, [0, 1])
    p_jt = create_p_i(Jt, [0, 1])
    pt_Ijt0 = get_pt_I_given_jt(I, p_i, Ci, It, Jt, 0, p_it, p_jt)
    pt_Ijt1 = get_pt_I_given_jt(I, p_i, Ci, It, Jt, 1, p_it, p_jt)

    p_Ij = {k: v for k, v in sorted(p_Ij.items(), key=lambda item: item[0])}
    pt_Ijt0 = {k: v for k, v in sorted(pt_Ijt0.items(), key=lambda item: item[0])}
    pt_Ijt1 = {k: v for k, v in sorted(pt_Ijt1.items(), key=lambda item: item[0])}

    D0 = kl_divergence(p_Ij.values(), pt_Ijt0.values())
    D1 = kl_divergence(p_Ij.values(), pt_Ijt1.values())

    if D0 < D1:  # we want to output the cluster associated with the smaller KL divergence
        return 0
    elif D1 < D0:
        return 1
    else:  # if they are equal, just choose at random
        return random.randint(0, 1)


def find_shared_argmin(I, p_i, Ci, It, J, p_j, Cj, Jt, K_i, K_j, K_it, K_jt, p_k, q_k, k, LAMBDA):
    """
    See equation 16 in STC paper, or Sarah's paper.
    Outputs best cluster (0 or 1) for value k.
    """
    # I = X, J = Y, K = Z
    # C_z(z) = argmin(zt in Zt) p(z)D(p(X|z)||pt(X|zt))
    #           + lambda q(z)D(q(Y|z)||qt(Y|zt))
    pk = p_k[k]
    p_Ik = get_p_I_given_j(I, K_i, k)
    p_it = create_p_i(It, [0, 1])
    p_kit = create_p_i(K_it, [0, 1])
    pt_Ikt0 = get_pt_I_given_jt(I, p_i, Ci, It, K_it, 0, p_it, p_kit)
    pt_Ikt1 = get_pt_I_given_jt(I, p_i, Ci, It, K_it, 1, p_it, p_kit)

    qk = q_k[k]
    p_jt = create_p_i(Jt, [0, 1])
    p_kjt = create_p_i(K_jt, [0, 1])
    q_Jk = get_p_I_given_j(J, K_j, k)
    qt_Jkt0 = get_pt_I_given_jt(J, p_j, Cj, Jt, K_jt, 0, p_jt, p_kjt)
    qt_Jkt1 = get_pt_I_given_jt(J, p_j, Cj, Jt, K_jt, 1, p_jt, p_kjt)

    p_Ik = {k: v for k, v in sorted(p_Ik.items(), key=lambda item: item[0])}
    pt_Ikt0 = {k: v for k, v in sorted(pt_Ikt0.items(), key=lambda item: item[0])}
    pt_Ikt1 = {k: v for k, v in sorted(pt_Ikt1.items(), key=lambda item: item[0])}
    q_Jk = {k: v for k, v in sorted(q_Jk.items(), key=lambda item: item[0])}
    qt_Jkt0 = {k: v for k, v in sorted(qt_Jkt0.items(), key=lambda item: item[0])}
    qt_Jkt1 = {k: v for k, v in sorted(qt_Jkt1.items(), key=lambda item: item[0])}

    # print(K_jt)
    # print(qt_Jkt1)
    D0 = (pk * kl_divergence(p_Ik.values(), pt_Ikt0.values())) + \
         (LAMBDA * qk * kl_divergence(q_Jk.values(), qt_Jkt0.values()))
    D1 = (pk * kl_divergence(p_Ik.values(), pt_Ikt1.values())) + \
         (LAMBDA * qk * kl_divergence(q_Jk.values(), qt_Jkt1.values()))

    if D0 < D1:  # we want to output the cluster associated with the smaller KL divergence
        return 0
    elif D1 < D0:
        return 1
    else:  # if they are equal, just choose at random
        return random.randint(0, 1)


def calculate_accuracy(I, Ci, I_labels):
    """
    Given: a list of examples, the clusters those values are in, and the labels associated with those examples
    Output: accuracy of clustering, i.e. examples with the same label are in the same cluster,
            and examples with opposite labels are in the opposite cluster.
    """
    accuracy = 0
    for example in range(len(I)):
        i = I[example]
        pred = find_associated_cluster(i, Ci)
        actual_label = I_labels[example]
        # arbitrary choice - saying that the prediction (0 or 1) should be the same as the cluster (0 or 1)
        # doesn't really matter since we end up returning max(accuracy, 1 - accuracy)
        if pred == actual_label:
            accuracy = accuracy + 1
    accuracy = accuracy / len(I)
    return max(accuracy, 1 - accuracy)

