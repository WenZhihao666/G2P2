import random
import numpy as np

def multitask_data_generator(labels, labeled_node_list, select_array, k_spt, k_val, k_qry, n_way):
    labels_local = labels  

    class_idx_list = []
    train_class_list = []
    val_class_list = []
    test_class_list = []
    for i in range(len(select_array)):
        class_idx_list.append([])
        train_class_list.append([])
        val_class_list.append([])
        test_class_list.append([])

    for j in labeled_node_list:
        for i in range(len(select_array)):
            if (labels_local[j] == select_array[i]):
                class_idx_list[i].append(j)

    usable_labels = []
    for i in range(len(class_idx_list)):
        if len(class_idx_list[i]) >= 30:
            usable_labels.append(i)

    random.shuffle(usable_labels)
    task_list = []
    for i in range(len(usable_labels) // n_way):
        task_idx = usable_labels[i * n_way:(i + 1) * n_way]
        task_list.append(task_idx)

    for i in range(len(select_array)):
        if i not in set(usable_labels):
            continue
        train_class_list[i] = np.random.choice(class_idx_list[i], k_spt, replace=False).tolist()
        val_class_temp = [n1 for n1 in class_idx_list[i] if n1 not in train_class_list[i]]
        val_class_list[i] = np.random.choice(val_class_temp, k_val, replace=False).tolist()
        test_class_temp = [n1 for n1 in class_idx_list[i] if
                           (n1 not in train_class_list[i]) and (n1 not in val_class_list[i])]
        test_class_list[i] = test_class_temp

    train_idx = []
    test_idx = []
    val_idx = []

    for i in range(len(task_list)):
        train_idx.append([])
        test_idx.append([])
        val_idx.append([])
        # print(task_list[i])
        for j in task_list[i]:
            train_idx[i] += train_class_list[j]
            val_idx[i] += val_class_list[j]
            test_idx[i] += test_class_list[j]

    return task_list, train_idx, val_idx, test_idx


