from surf.script_tab import keytab
import os, json, time, re, codecs
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging.handlers
import pandas as pd
import itertools
import numpy as np
# import kerastuner as kt


def pipe_train(dataobj, paras={}):
    outdata = dataobj
    return outdata


# class CVTuner(kt.engine.tuner.Tuner):
#     def run_trial(self, trial, X, y, splits, batch_size=32, epochs=1, callbacks=None):
#         val_losses = []
#         for train_indices, test_indices in splits:
#             X_train, X_test = [x[train_indices] for x in X], [x[test_indices] for x in X]
#             y_train, y_test = [a[train_indices] for a in y], [a[test_indices] for a in y]
#             if len(X_train) < 2:
#                 X_train = X_train[0]
#                 X_test = X_test[0]
#             if len(y_train) < 2:
#                 y_train = y_train[0]
#                 y_test = y_test[0]
#
#             model = self.hypermodel.build(trial.hyperparameters)
#             hist = model.fit(X_train, y_train,
#                              validation_data=(X_test, y_test),
#                              epochs=epochs,
#                              batch_size=batch_size,
#                              callbacks=callbacks)
#
#             val_losses.append([hist.history[k][-1] for k in hist.history])
#         val_losses = np.asarray(val_losses)
#         self.oracle.update_trial(trial.trial_id,
#                                  {k: np.mean(val_losses[:, i]) for i, k in enumerate(hist.history.keys())})
#         self.save_model(trial.trial_id, model)


train_func = {
    "训练": pipe_train,
    "xgboost": None,
    "tcn": None,
    "tabnet": None,
}
