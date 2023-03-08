
import json
import pandas as pa
import numpy as np


class Classifier:

    def __init__(self, path, prior = []):
        self.content = json.load(open(path, "r"))

        if self.content["parameters"]["loss"] in ["L1", "L2"]:
            raise Exception(path + " is a regressor, do not use \"Classifier\"")



    #def likelihood(self, thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]):
    #def likelihood(self, thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]):
    def likelihood(self, thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]):
        P_likeli = pa.DataFrame()
        N_likeli = pa.DataFrame()

        P_likeli["threshold"] = thresholds
        N_likeli["threshold"] = thresholds

        for label in self.content["parameters"]["output labels"]:
            P_tmp = []
            N_tmp = []

            for t in thresholds:
                P_count = 0
                N_count = 0

                positive = 0
                negative = 0

                for i in range(len(self.content["test data"][label]["expected"])):
                    if self.content["test data"][label]["expected"][i] == 1:
                        P_count += 1
                        if self.content["test data"][label]["predicted"][i] >= t:
                            positive += 1
                    else:
                        N_count += 1
                        if self.content["test data"][label]["predicted"][i] < t:
                            negative += 1

                P_tmp.append(positive / P_count)
                N_tmp.append(negative / N_count)

            P_likeli[label] = P_tmp
            N_likeli[label] = N_tmp

        return P_likeli, N_likeli



    def prior(self):
        prior = pa.DataFrame()

        for label in self.content["parameters"]["output labels"]:
            prior[label] = [self.content["test data"][label]["expected"].count(1) / len(self.content["test data"][label]["expected"])]

        return prior



    def posterior(self, P, N, prior):
        P_posterior = pa.DataFrame()
        N_posterior = pa.DataFrame()
        P_evidence = pa.DataFrame()
        N_evidence = pa.DataFrame()

        P_posterior["threshold"] = P["threshold"]
        N_posterior["threshold"] = P["threshold"]
        P_evidence["threshold"] = P["threshold"]
        N_evidence["threshold"] = P["threshold"]

        for label in self.content["parameters"]["output labels"]:
            P_evidence[label] = np.array(P[label]) * prior[label][0] + (1-np.array(N[label]))*(1-np.array(prior[label][0]))
            N_evidence[label] = np.array(N[label]) * (1-prior[label][0]) + (1-np.array(P[label]))*np.array(prior[label][0])
            P_posterior[label] = np.array(P[label]) * prior[label][0] / np.array(P_evidence[label])
            N_posterior[label] = np.array(N[label]) * (1-prior[label][0]) / np.array(N_evidence[label])

        return P_posterior, N_posterior, P_evidence, N_evidence






class Regressor():

    def __init__():
        self.content = json.load(open(path, "r"))
        if content["parameters"][loss] not in ["L1", "L2"]:
            raise Exception(path + " is a classifier, do not use \"Regressor\"")

    def preprocess():
        pass