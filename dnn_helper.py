import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import shutil
import warnings

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.callbacks import Callback
from keras.losses import binary_crossentropy
from keras.models import model_from_json
from keras import backend as K

from plot_utils import plot_roc_auc


def KFolds_flow_from_dataframe(dataframe, generators, kfolds=10, directory=None, x_col='filename', y_col='class',
                               weight_col=None, target_size=(256, 256), color_mode='rgb', classes=None,
                               class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None,
                               save_prefix='', save_format='png', interpolation='nearest', validate_filenames=True):
    train_gens = []
    valid_gens = []
    Kfolds_gens = []

    stratify_df = dataframe[y_col] if shuffle else None
    remain_df, kfold_df = train_test_split(dataframe, test_size=1 / (kfolds), stratify=stratify_df,
                                           shuffle=shuffle, random_state=seed)

    train_gens.append(
        generators[0].flow_from_dataframe(kfold_df, directory, x_col, y_col, weight_col, target_size, color_mode,
                                          classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix,
                                          save_format, None, interpolation, validate_filenames)
    )

    i = 1
    while i < kfolds - 1:
        try:
            stratify_df = remain_df[y_col] if shuffle else None
            remain_df, kfold_df = train_test_split(remain_df, test_size=1 / (kfolds - i), stratify=stratify_df,
                                                   shuffle=shuffle, random_state=seed)
        except ValueError as e:
            print(f"Stratify is not posible at kfold {i} due to: {e}")
            remain_df, kfold_df = train_test_split(remain_df, test_size=1 / (kfolds - i), shuffle=shuffle, random_state=seed)

        train_gens.append(
            generators[0].flow_from_dataframe(kfold_df, directory, x_col, y_col, weight_col, target_size, color_mode,
                                              classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix,
                                              save_format, None, interpolation, validate_filenames)
        )
        valid_gens.append(
            generators[1].flow_from_dataframe(kfold_df, directory, x_col, y_col, weight_col, target_size, color_mode,
                                              classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix,
                                              save_format, None, interpolation, validate_filenames)
        )

        i = i + 1
    remain_gen = generators[1].flow_from_dataframe(remain_df, directory, x_col, y_col, weight_col, target_size,
                                                   color_mode, classes, class_mode, batch_size, shuffle, seed,
                                                   save_to_dir, save_prefix, save_format, None, interpolation,
                                                   validate_filenames)
    valid_gens.append(remain_gen)

    i = 0
    while i < kfolds - 1:
        Kfolds_gens.append((train_gens[i], valid_gens[i]))
        i = i + 1

    return Kfolds_gens


def fold_training(KFolds_gens, k, model, optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'],
                  class_weight=None, models_dir='', callbacks_list=None, history_callback=None, plot_history=False):
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    train_gen = KFolds_gens[k][0]
    valid_gen = KFolds_gens[k][1]
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=train_gen.n // train_gen.batch_size,
                                  epochs=1,
                                  validation_data=valid_gen,
                                  validation_steps=valid_gen.n // valid_gen.batch_size,
                                  class_weight=class_weight,
                                  # use_multiprocessing = True,
                                  # workers = 2 * multiprocessing.cpu_count(),
                                  callbacks=callbacks_list)

    if history_callback is not None:
        history = history_callback

    save_train(model, history, models_dir)
    if plot_history:
        visualize_training(history.history)

    return model, history


def config_model_trainable(model, config, last_block=0, base_model=None):
    if not config in ['full', 'partial', 'top']:
        raise ValueError(
            f"config value is {config} parameter should be one of the following values: 'full','partial' or 'top'")

    trainable_reference = config == 'full'

    # train only the top layers (which were randomly initialized)
    # i.e. freeze all layers of the based model that is already pre-trained.
    if config == 'top':
        if base_model is None:
            raise ValueError("config value is {config}, but base model have been not passed on function")

        for layer in base_model.layers:
            layer.trainable = False

    # if config = 'partial' train only the layers after last_block
    # otherwise train all the layers
    else:
        if type(last_block) is int:
            for layer in model.layers[:last_block]:
                layer.trainable = trainable_reference
            for layer in model.layers[last_block:]:
                layer.trainable = True
        else:
            set_trainable = trainable_reference
            for layer in model.layers:
                if layer.name == last_block:
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, target='target', shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    if target is not None:
        labels = dataframe.pop(target)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def dnn_plot_roc_auc(X, y_test, model, feature_selection, batch_size):
    x_ds = df_to_dataset(X[feature_selection], target=None, shuffle=False, batch_size=batch_size)
    plot_roc_auc(y_test, model.predict(x_ds))


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2.0 * K.sum(intersection) + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )
    return 1.0 - score


def bce_dice_loss(y_true, y_pred, clip_loss = None):
    if clip_loss is None:
        return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    else:
        return tf.clip_by_value(binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred), -clip_loss, clip_loss)

def make_pred(model, gen, steps = None):
    if steps is None:
        steps = gen.n // gen.batch_size
    y_pred = model.predict_generator(
            gen,
            steps=steps,
            verbose=1,
        )
    
    if type(y_pred[0]) is list:
        y_pred = np.array([y.reshape(-1) for y in y_pred])
    
    return y_pred

def save_train(model, history=None, models_dir="", json_path="", weights_path="", save_weights=True, history_path="", feature_selection=None, feature_selection_path=""):
   
    if models_dir != "" and models_dir[-1] != "/" and models_dir[-1] != r"\ ".strip():
        models_dir = models_dir + "/"
    
    # Save model structure on JSON
    if json_path == "":
        json_path = models_dir + "model.json"
    # serialize model to JSON
    try:
        model_json = model.to_json()
        with open(json_path, "w") as json_file:
            json_file.write(model_json)   
    except NotImplementedError as e:
        print("Model JSON could not be save due to: ", e)

    # Save weights
    if save_weights:
        if weights_path == "":
            weights_path = models_dir + "model_weights"
        # serialize weights to HDF5
        model.save_weights(weights_path)

    # Save history
    if history is None:
        history = model.history.history
    else:
        for key in history:
            history[key] = history[key] + model.history.history.get(key,[])
    
    if history_path == "":
        history_path = models_dir + "model_history.pickle"
    with open(history_path, 'wb') as pickle_file:
        pickle.dump(history, pickle_file)
    
    # Save feature selection
    if feature_selection is not None:
        if feature_selection_path == "":
            feature_selection_path = models_dir + "feature_selection.pickle"
        
        with open(feature_selection_path, 'wb') as pickle_file:
            pickle.dump(feature_selection, pickle_file)

def load_train(models_dir=''):
    # load json and create model
    with open(models_dir + "model.json", "r") as json_file:
        loaded_model_json = json_file.read()
        
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(models_dir +  "model_weights.h5")
    print("Loaded model and history")
    
    try:
        with open(models_dir + "history.pickle", "wb") as file:
            history = pickle.load(file)
    except:
        print("Train history could not be loaded")
        history = None
    return loaded_model, history


class BatchHistoryEarlyStopping(Callback):
    def __init__(
            self,
            valid_generator,
            targets,
            batch_freq=100,
            reset_on_train=True,
            early_stopping=False,
            monitor="val_loss",
            min_delta=0,
            patience=0,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
    ):

        super().__init__()
        self.valid_generator = valid_generator
        self.targets = targets
        self.batch_freq = batch_freq
        self.reset_on_train = reset_on_train
        self.early_stopping = early_stopping
        if self.early_stopping:
            self.monitor = monitor
            self.baseline = baseline
            self.patience = patience
            self.verbose = verbose
            self.min_delta = min_delta
            self.wait = 0
            self.stopped_batch = 0
            self.restore_best_weights = restore_best_weights
            self.best_weights = None

            if mode not in ["auto", "min", "max"]:
                warnings.warn(
                    "EarlyStopping mode %s is unknown, "
                    "fallback to auto mode." % mode,
                    RuntimeWarning,
                )
                mode = "auto"

            if mode == "min":
                self.monitor_op = np.less
            elif mode == "max":
                self.monitor_op = np.greater
            else:
                if "acc" in self.monitor:
                    self.monitor_op = np.greater
                else:
                    self.monitor_op = np.less

            if self.monitor_op == np.greater:
                self.min_delta *= 1
            else:
                self.min_delta *= -1

    def on_train_begin(self, logs={}):
        if self.reset_on_train:
            self.reset_stats()

    def on_batch_end(self, batch, logs={}):
        if batch != 0 and batch % self.batch_freq == 0:
            # Save validation metrics
            valid_metrics = self.model.evaluate_generator(self.valid_generator)
            for i, value in enumerate(valid_metrics):
                self.history[self.metrics[i]] = self.history[self.metrics[i]] + [value]

            # Save training metrics
            i = len(self.metrics) // 2
            while i < len(self.metrics):
                self.history[self.metrics[i]] = self.history[self.metrics[i]] + [
                    logs[self.metrics[i]]
                ]
                i = i + 1

            if self.early_stopping:
                current = self.get_monitor_value(self.history)
                if current is None:
                    return

                if self.monitor_op(current - self.min_delta, self.best):
                    self.best = current
                    self.wait = 0
                    if self.restore_best_weights:
                        self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_batch = batch
                        self.model.stop_training = True
                        if self.restore_best_weights:
                            if self.verbose > 0:
                                print(
                                    "Restoring model weights from the end of "
                                    "the best epoch"
                                )
                            self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.early_stopping and self.stopped_epoch > 0 and self.verbose > 0:
            print("Batch %05d: early stopping" % (self.stopped_batch + 1))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s"
                % (self.monitor, ",".join(list(logs.keys()))),
                RuntimeWarning,
            )
        return monitor_value[-1]

    def reset_stats(self):
        self.metrics = (
                ["loss"]
                + ["pred_" + target + "_loss" for target in self.targets]
                + ["pred_" + target + "_accuracy" for target in self.targets]
        )
        self.metrics = ["val_" + metric for metric in self.metrics] + self.metrics
        metrics_values = [[]] * len(self.metrics)
        self.history = dict(zip(self.metrics, metrics_values))

        if self.early_stopping:
            # Allow instances to be re-used
            self.wait = 0
            self.stopped_epoch = 0
            if self.baseline is not None:
                self.best = self.baseline
            else:
                self.best = np.Inf if self.monitor_op == np.less else -np.Inf


def visualize(history, key, y_label, x_label="epoch", title=None):
    plt.plot(history[key])
    plt.plot(history["val_" + key])
    if title is None:
        title = "model_" + key
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(["train", "valid"], loc="upper left")
    plt.show()


def visualize_acc_loss(history, output_name="", only_loss=False):
    if output_name != "":
        output_name = output_name + "_"
    if not only_loss:
        try:
            metric = "acc"
            visualize(history, output_name + metric, metric)
        except KeyError:
            metric = "accuracy"
            visualize(history, output_name + metric, metric)
    metric = "loss"
    visualize(history, output_name + metric, metric)


def visualize_training(history):
    if len(history.keys()) <= 2:
        visualize_acc_loss(history)
    else:
        for key in history:
            if key == "loss":
                visualize_acc_loss(history, only_loss=True)
            elif key[:4] != "val_" and key[-5:] == "_loss":
                output_name = key[:-5]
                visualize_acc_loss(history, output_name=output_name)
                
class Model_generator:
    def __init__(self, input_shape, n_outputs, n_units, model_type='dnn_baseline', activation='sigmoid', lstm_blocks=1, dropout_rate=0, recurrent_dropout=0, lstm_l1=0, lstm_l2=0, droput_input_cols=None, remain_input_cols=None):
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.n_units = n_units
        self.model_type = model_type
        self.model_type_list = ['dnn_baseline', 'lstm_baseline', 'lstm', 'attention_lstm', 'attention_lstm_residual', 'attention_lstm_dropout_input']
        self.activation = activation
        self.lstm_blocks = lstm_blocks
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.lstm_l1 = lstm_l1
        self.lstm_l2 = lstm_l2
        self.droput_input_cols = droput_input_cols
        self.remain_input_cols = remain_input_cols

    def get_model(self, model_type=None):
        if model_type is None:
            model_type = self.model_type
        if model_type not in self.model_type_list:
            raise ValueError(f'Model type {model_type} not in list. Choose one of the following model types: {self.model_type_list}')
            
        if model_type == 'dnn_baseline':
            return self.dnn_baseline()
        elif model_type == 'lstm_baseline':
            return self.lstm_baseline()
        elif model_type == 'lstm':
            return self.lstm()
        elif model_type == 'attention_lstm':
            return self.attention_lstm()
        elif model_type == 'attention_lstm_residual':
            return self.attention_lstm_residual()
        elif model_type == 'attention_lstm_dropout_input':
            return self.attention_lstm_dropout_input()
        else:
            print("Implementation error, model_type {model_type} is missing")

    def dnn_baseline(self):
        input_x = Input(shape = self.input_shape)
        X = input_x
        X = Dense(self.n_outputs)(X)
        X = Activation(self.activation, name = 'output')(X)

        return Model(inputs=input_x, outputs=X)
    
    def lstm_baseline(self):
        input_x = Input(shape = self.input_shape)
        X = input_x
        X = BatchNormalization()(X)
        
        X = LSTM(self.n_units, return_sequences = False, 
                 recurrent_dropout=self.recurrent_dropout, 
                 kernel_regularizer=l1_l2(self.lstm_l1, self.lstm_l2),
                 activity_regularizer=l1_l2(self.lstm_l1, self.lstm_l2))(X)
        if self.dropout_rate > 0:
            X = Dropout(self.dropout_rate)(X)
        X = BatchNormalization()(X)
        X = Dense(self.n_outputs)(X)
        X = Activation(self.activation, name = 'output')(X)

        return Model(inputs=input_x, outputs=X)
    
    def lstm(self):
        input_x = Input(shape = self.input_shape)
        X = input_x
        X = BatchNormalization()(X)
        for i in range(self.lstm_blocks):
            X = LSTM(self.n_units, return_sequences = True, 
                 recurrent_dropout=self.recurrent_dropout, 
                 kernel_regularizer=l1_l2(self.lstm_l1, self.lstm_l2),
                 activity_regularizer=l1_l2(self.lstm_l1, self.lstm_l2))(X)
            if self.dropout_rate > 0:
                X = Dropout(self.dropout_rate)(X)
            X = BatchNormalization()(X)
                
        X = LSTM(self.n_units, return_sequences = False, 
                 recurrent_dropout=self.recurrent_dropout, 
                 kernel_regularizer=l1_l2(self.lstm_l1, self.lstm_l2),
                 activity_regularizer=l1_l2(self.lstm_l1, self.lstm_l2))(X)
        if self.dropout_rate > 0:
            X = Dropout(self.dropout_rate)(X)
        X = BatchNormalization()(X)
        X = Dense(self.n_outputs)(X)
        X = Activation(self.activation, name = 'output')(X)

        return Model(inputs=input_x, outputs=X)
    
    def attention_lstm(self):
        input_x = Input(shape = self.input_shape, name = 'input')
        X = input_x
        
        for i in range(self.lstm_blocks):
            query = Dense(10, name='query_' + str(i))(X)
            key = Dense(10, name='key_' + str(i))(X)
            attention_weights = AdditiveAttention(use_scale = False, name='attention_'+str(i))([query, X, key])
            attention_weights = Dense(1, activation='softmax', name='attention_weights_'+str(i))(attention_weights)
            context = Multiply(name='context_'+str(i))([attention_weights,X])
            X = LSTM(self.n_units, return_sequences = True, 
                     recurrent_dropout=self.recurrent_dropout, 
                     kernel_regularizer=l1_l2(self.lstm_l1, self.lstm_l2),
                     activity_regularizer=l1_l2(self.lstm_l1, self.lstm_l2),
                     name = 'lstm_' + str(i))(context)
            if self.dropout_rate > 0:
                X = Dropout(self.dropout_rate, name='dropout_'+str(i))(X)
                
        X = LSTM(self.n_units, return_sequences = False, 
                 recurrent_dropout=self.recurrent_dropout, 
                 kernel_regularizer=l1_l2(self.lstm_l1, self.lstm_l2),
                 activity_regularizer=l1_l2(self.lstm_l1, self.lstm_l2),
                 name = 'lstm_last')(X)
        if self.dropout_rate > 0:
            X = Dropout(self.dropout_rate, name='dropout_last')(X)
        X = Dense(self.n_outputs, activation=self.activation, name = 'output')(X)

        return Model(inputs=input_x, outputs=X, name='attention_lstm')
    
    def attention_lstm_residual(self):
        input_x = Input(shape = self.input_shape, name = 'input')
        X = input_x
        
        for i in range(self.lstm_blocks):
            query = Dense(10, name='query_' + str(i))(X)
            key = Dense(10, name='key_' + str(i))(X)
            attention_weights = AdditiveAttention(use_scale = False, name='attention_'+str(i))([query, X, key])
            attention_weights = Dense(1, activation='softmax', name='attention_weights_'+str(i))(attention_weights)
            context = Multiply(name='context_'+str(i))([attention_weights,X])
            X = LSTM(self.n_units, return_sequences = True, 
                     recurrent_dropout=self.recurrent_dropout, 
                     kernel_regularizer=l1_l2(self.lstm_l1, self.lstm_l2),
                     activity_regularizer=l1_l2(self.lstm_l1, self.lstm_l2),
                     name = 'lstm_' + str(i))(context)
            if self.dropout_rate > 0:
                X = Dropout(self.dropout_rate, name='dropout_'+str(i))(X)
                
        X = LSTM(self.n_units, return_sequences = False, 
                 recurrent_dropout=self.recurrent_dropout, 
                 kernel_regularizer=l1_l2(self.lstm_l1, self.lstm_l2),
                 activity_regularizer=l1_l2(self.lstm_l1, self.lstm_l2),
                 name = 'lstm_last')(X)
        if self.dropout_rate > 0:
            X = Dropout(self.dropout_rate, name='dropout_last')(X)
        
        crop_input = Cropping1D(cropping=(0, self.input_shape[0] - 1), name='crop_input')(input_x)
        if self.dropout_rate > 0:
            crop_input = Dropout(self.dropout_rate, name='dropout_crop_input')(crop_input)
        flatten_crop = Flatten()(crop_input)
        query_input = Dense(10, name='query_input')(flatten_crop)
        key_input = Dense(10, name='key_input')(flatten_crop)
        attention_weights_input = AdditiveAttention(use_scale = False, name='attention_input')([query_input, flatten_crop, key_input])
        attention_weights_input = Dense(1, activation='softmax', name='attention_weights_input')(attention_weights_input)
        context_input = Multiply(name='context_input')([attention_weights_input, flatten_crop])
        concat = Concatenate()([X, context_input])
        X = Dense(self.n_outputs, activation=self.activation, name = 'output')(concat)

        return Model(inputs=input_x, outputs=X, name='attention_lstm')
    
    def attention_lstm_dropout_input(self):
        dropout_input = Input(shape = (seq_len, droput_input_cols), name = 'dropout_input')
        remain_input = Input(shape = (seq_len, remain_input_cols), name = 'remain_input')

        dropout_x = Dropout(self.dropout_rate)(dropout_input)

        X = Concatenate(axis=-1)([remain_input, dropout_x])
        
        for i in range(self.lstm_blocks):
            query = Dense(10)(X)
            key = Dense(10)(X)
            context = AdditiveAttention()([query, X, key])
            #context = one_step_attention(a)
            X = LSTM(self.n_units, return_sequences = True, recurrent_dropout=self.recurrent_dropout)(context)
            if self.dropout_rate > 0:
                X = Dropout(self.dropout_rate)(X)
                
        X = LSTM(self.n_units, return_sequences = False, recurrent_dropout=self.recurrent_dropout)(X)
        if self.dropout_rate > 0:
            X = Dropout(self.dropout_rate)(X)
        X = Dense(self.n_outputs)(X)
        X = Activation(self.activation, name = 'output')(X)

        return Model(inputs=[dropout_input, remain_input], outputs=X)