import os
import argparse
import sys
import re
import time
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import cv2

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


import warnings
warnings.filterwarnings("ignore")


def get_x_features_array_and_df(keypoints):

    feat_dict = {}

    for i in range(0, 18):
        feat_dict["bp_" + str(i) + '_x'] = None
        feat_dict["bp_" + str(i) + '_y'] = None

    for i in range(0, len(keypoints)-1):
        bps = int(re.findall(r'\d+', keypoints[i].split("(")[0])[-1])
        coordinates = keypoints[i].split("(")[1].split(")")[0].split(",")

        #print(i, bps)
        feat_dict["bp_" + str(bps) + '_x'] = float(coordinates[0])
        feat_dict["bp_" + str(bps) + '_y'] = float(coordinates[1])

#     for k, v in feat_dict.items():
#         print(k, v)

    df = pd.DataFrame(feat_dict.items()).set_index(0).T
    return list(feat_dict.values()), df


def inference(image, fps_time, out=None):
    humans = e.inference(image,
                         resize_to_default=(w > 0 and h > 0),
                         upsample_size=4.0)

    pred = None
    if len(humans) > 0:
        keypoints = str(str(str(humans[0]).split('BodyPart:')[
                        1:]).split('-')).split(' score=')

        arr, fdf = get_x_features_array_and_df(keypoints)
        pred = tree_clf.predict(fdf.fillna(0))
        print(pred)

        bpred = bag_clf.predict(fdf.fillna(0))
        print(bpred)

        rnd_clf
        rnd_pred = rnd_clf.predict(fdf.fillna(0))
        print(rnd_pred)

        yhat = lr_model.predict(fdf.fillna(0))
        print(yhat)

        labels = kmeans.predict(fdf.fillna(0))
        print(labels)

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    if pred:
        cv2.putText(image, "single tree: "+pred[0], (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, "bagging: "+bpred[0], (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, "random forest: "+rnd_pred[0], (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, "logistic reg: "+yhat[0], (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, "kmeans cluster: "+str(labels[0]), (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if out:
        out.write(image)

    cv2.imshow('Frame', image)
    fps_time = time.time()

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human pose classification')
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    path = args.file

    df = pd.read_csv('data.csv')

    feature_names = ['nose_x', 'nose_y', 'neck_x', 'neck_y', 'Rshoulder_x', 'Rshoulder_y',
                     'Relbow_x', 'Relbow_y', 'Rwrist_x', 'RWrist_y', 'LShoulder_x',
                     'LShoulder_y', 'LElbow_x', 'LElbow_y', 'LWrist_x', 'LWrist_y', 'RHip_x',
                     'RHip_y', 'RKnee_x', 'RKnee_y', 'RAnkle_x', 'RAnkle_y', 'LHip_x',
                     'LHip_y', 'LKnee_x', 'LKnee_y', 'LAnkle_x', 'LAnkle_y', 'REye_x',
                     'REye_y', 'LEye_x', 'LEye_y', 'REar_x', 'REar_y', 'LEar_x', 'LEar_y']

    x = df[feature_names]
    y = df['class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

    # Single decision tree
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(x_train, y_train)
    y_pred_tree = tree_clf.predict(x_test)

    # Bagging classifier
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42),
        n_estimators=500,
        max_samples=100,
        bootstrap=True,
        n_jobs=-1,
        random_state=42)

    bag_clf.fit(x_train, y_train)
    y_pred = bag_clf.predict(x_test)

    # Random forest
    rnd_clf = RandomForestClassifier(
        n_estimators=500, max_leaf_nodes=5, n_jobs=-1, random_state=42)
    rnd_clf.fit(x_train, y_train)
    y_pred_rf = rnd_clf.predict(x_test)

    # Logistic regression
    lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    # define the model evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(
        lr_model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report the models performance
    print()
    print("single tree: ", accuracy_score(y_test, y_pred_tree))
    print("ensemble bagging: ", accuracy_score(y_test, y_pred))
    print("random forest: ", accuracy_score(y_test, y_pred_rf))
    print('logistic regression %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    lr_model.fit(x, y)

    # K means
    # nc = range(1, 20)
    # kmeans = [KMeans(n_clusters=i) for i in nc]
    # score = [kmeans[i].fit(x_train).score(x_train) for i in range(len(kmeans))]
    # plt.plot(nc, score)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Score')
    # plt.title('Elbow Curve')
    # plt.show()

    kmeans = KMeans(n_clusters=6).fit(x)

    model = 'cmu'

    resolution = '864x736'
    w, h = model_wh(resolution)
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    file_type = path.split('.')[-1]
    if file_type == 'avi' or file_type == 'mp4':

        cap = cv2.VideoCapture(path)
        cap.set(3, 640)
        cap.set(4, 480)

        if cap.isOpened() is False:
            print("Error opening video stream or file")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0,
                              (int(cap.get(3)), int(cap.get(4))))

        fcount = 0
        frames = []
        while True:
            ret_val, image = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            fcount += 1

            print("frame no : ", fcount)

            if not ret_val:
                break

            image = inference(image, time.time())

            cv2.imwrite(f'output/{fcount}_{path}.jpg', image)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            if len(frames) > 200:
                break

        print("{} frames processed".format(len(frames)))
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    elif file_type == 'jpg' or file_type == 'png':

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = inference(image, time.time())

        cv2.waitKey(0)
