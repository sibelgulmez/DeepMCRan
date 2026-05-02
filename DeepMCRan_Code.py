import random
import numpy as np
from sklearn.model_selection import train_test_split
import os, re
from sklearn.metrics import confusion_matrix

from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate,
    Dense, Dropout, Add, BatchNormalization, Activation, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model

def residual_block_merge(x, filters, block_name):
    shortcut = x
    x = Dense(filters, activation='relu', name=f'{block_name}_Dense1')(x)
    x = BatchNormalization(name=f'{block_name}_BN1')(x)
    x = Dense(filters, activation=None, name=f'{block_name}_Dense2')(x)
    x = BatchNormalization(name=f'{block_name}_BN2')(x)

    if shortcut.shape[-1] != filters:
        shortcut = Dense(filters, activation=None, name=f'{block_name}_Shortcut')(shortcut)

    x = Add(name=f'{block_name}_Add')([x, shortcut])
    x = Activation('relu', name=f'{block_name}_ReLU')(x)
    return x

def textcnn_block(inp, vocab_size, embedding_dim, name_prefix):
    emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name=f'{name_prefix}_Embedding')(inp)

    convs = []
    for k in [3, 4, 5]:
        c = Conv1D(filters=64, kernel_size=k, activation='relu', padding='same', name=f'{name_prefix}_Conv_{k}')(emb)
        p = GlobalMaxPooling1D(name=f'{name_prefix}_MaxPool_{k}')(c)
        convs.append(p)

    merged = Concatenate(name=f'{name_prefix}_MergedConv')(convs)
    return merged

def textcnn_residual(input_lengths, vocab_size, embedding_dim=32):
    channel_names = ['Opcode', 'APICall', 'DLL', 'Mutex']
    inputs, outputs = [], []

    for name, seq_len in zip(channel_names, input_lengths):
        inp = Input(shape=(seq_len,), name=f'{name}_Input')
        out = textcnn_block(inp, vocab_size, embedding_dim, name_prefix=name)
        inputs.append(inp)
        outputs.append(out)

    merged = Concatenate(name='Merged_Features')(outputs)
    merged = Dropout(0.3, name='Merged_Dropout')(merged)

    x = residual_block_merge(merged, 128, "Merged_Residual")
    x = Dropout(0.3, name='PostResidual_Dropout')(x)
    x = Dense(64, activation='relu', name='Dense_64')(x)
    x = Dense(32, activation='relu', name='Dense_32')(x)

    output = Dense(1, activation='sigmoid', name='Output')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()
    return model

def padding(data, max_len):
    if len(data)<max_len:
        for i in range(len(data), max_len):
            data.append("NONE")
    return data[:max_len]

def readData (ransomware_dataset_path, benign_dataset_path, featureName, featureLength):
    ransomware_fileList = sorted(os.listdir(ransomware_dataset_path))
    benign_fileList  = sorted(os.listdir(benign_dataset_path))
    data = []
    labels = []
    for i in range(9000):
        rw_file = ransomware_fileList[i]
        filePath = os.path.join(ransomware_dataset_path, featureName, rw_file)
        f = open(filePath, "r")
        line = f.read().replace("\n","")
        f.close()
        if line == "":
            sample = "NONE"
        else:
            sample = line.replace(",", " ")
        sample = (re.sub(r'[^a-zA-Z0-9\s]+', '', sample).replace("  ", " ")).split(" ")
        data.append(padding(sample, featureLength))
        labels.append(1)

        bn_file = benign_fileList[i]
        filePath = os.path.join(benign_dataset_path, featureName, bn_file)
        f = open(filePath, "r")
        line = f.read().replace("\n","")
        f.close()
        if line == "":
            sample = "NONE"
        else:
            sample = line.replace(",", " ")
        sample = (re.sub(r'[^a-zA-Z0-9\s]+', '', sample).replace("  ", " ")).split(" ")
        data.append(padding(sample, featureLength))
        labels.append(0)
    return data, labels

def getVocab (data1, data2, data3, data4):
    uniques = ["UNKNOWN"]
    for sample in data1:
        for s in sample:
            if s not in uniques:
                uniques.append(s)
    for sample in data2:
        for s in sample:
            if s not in uniques:
                uniques.append(s)
    for sample in data3:
        for s in sample:
            if s not in uniques:
                uniques.append(s)
    for sample in data4:
        for s in sample:
            if s not in uniques:
                uniques.append(s)
    return uniques

def integerEncoding (data, vocab):
    index = 0
    word2idx = {w: i for i, w in enumerate(vocab)}
    for sample in data:
        integer_form = []
        for s in sample:
            if s in vocab:
                integer_form.append(word2idx.get(s, 0))
            else:
                integer_form.append(0) # "unknown"
        data[index] = integer_form
        index += 1
    return data

def runner (ransomware_dataset_path, benign_dataset_path, featureLength1, featureLength2, featureLength3, featureLength4):
    data1, labels = readData(ransomware_dataset_path, benign_dataset_path, "opcode2000", featureLength1)
    data2, labels = readData(ransomware_dataset_path, benign_dataset_path, "apicall2500", featureLength2)
    data3, labels = readData(ransomware_dataset_path, benign_dataset_path, "dll100", featureLength3)
    data4, labels = readData(ransomware_dataset_path, benign_dataset_path, "mutex100", featureLength4)

    combined = list(zip(data1, data2, data3, data4, labels,))
    random.shuffle(combined)
    data1, data2, data3, data4, labels = zip(*combined)
    data1, data2, data3, data4, labels, = list(data1), list(data2), list(data3), list(data4), list(labels)
    X_train_data1, X_test_data1, X_train_data2, X_test_data2, X_train_data3, X_test_data3, X_train_data4, X_test_data4, y_train, y_test = train_test_split(data1, data2, data3, data4, labels, test_size=0.1, random_state=42)


    vocab = getVocab(X_train_data1, X_train_data2, X_train_data3, X_train_data4)
    vocab_size = len(vocab) + 1


    X_train_data1 = np.array(integerEncoding(X_train_data1, vocab), dtype = np.int32)
    X_train_data2 = np.array(integerEncoding(X_train_data2, vocab), dtype = np.int32)
    X_train_data3 = np.array(integerEncoding(X_train_data3, vocab), dtype = np.int32)
    X_train_data4 = np.array(integerEncoding(X_train_data4, vocab), dtype = np.int32)

    X_test_data1 = np.array(integerEncoding(X_test_data1, vocab), dtype = np.int32)
    X_test_data2 = np.array(integerEncoding(X_test_data2, vocab), dtype = np.int32)
    X_test_data3 = np.array(integerEncoding(X_test_data3, vocab), dtype = np.int32)
    X_test_data4 = np.array(integerEncoding(X_test_data4, vocab), dtype = np.int32)

    y_train = np.array(y_train, dtype = np.int32)
    y_test = np.array(y_test, dtype = np.int32)



    model = textcnn_residual([featureLength1, featureLength2, featureLength3, featureLength4], vocab_size, 32)

    indices = range(len(y_train))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    X1_train, X1_val = X_train_data1[train_idx], X_train_data1[val_idx]
    X2_train, X2_val = X_train_data2[train_idx], X_train_data2[val_idx]
    X3_train, X3_val = X_train_data3[train_idx], X_train_data3[val_idx]
    X4_train, X4_val = X_train_data4[train_idx], X_train_data4[val_idx]
    y_train_, y_val_ = y_train[train_idx], y_train[val_idx]


    model.fit(
        [X1_train, X2_train, X3_train, X4_train],
        y_train_,
        validation_data=([X1_val, X2_val, X3_val, X4_val], y_val_),
        epochs=10,
        batch_size=50,
        verbose=0
    )

    y_pred = model.predict([X_test_data1, X_test_data2, X_test_data3, X_test_data4])

    cfm = confusion_matrix(y_test, (y_pred > 0.5).astype(int))
    print(cfm)
    return