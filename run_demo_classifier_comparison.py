from __future__ import print_function

from uci_comparison import *
from sklearn.ensemble.forest import RandomForestClassifier
from InterpretableDecisionTreeClassifier import *

estimators = {
              'Random Forest': RandomForestClassifier(),
              'D.Tree3': DecisionTreeClassifier(max_depth=3),
              'Interpr.D.Tree3': IDecisionTreeClassifier(max_depth=3),
              'D.Tree5': DecisionTreeClassifier(max_depth=5),
              'Interpr.D.Tree5': IDecisionTreeClassifier(max_depth=5),
            }

# optionally, pass a list of UCI dataset identifiers as the datasets parameter, e.g. datasets=['iris', 'diabetes']
# optionally, pass a dict of scoring functions as the metric parameter, e.g. metrics={'F1-score': f1_score}
compare_estimators(estimators)

print("")
for d in comparison_datasets:
    # load
    try:
        X, y = getdataset(d)
    except:
        print("FAILED TO LOAD",d," - SKIPPING")
        continue
    # train
    clf = IDecisionTreeClassifier(max_depth=3).fit(X,y)
    # tostring
    itree = str(clf)
    # compare with full DT
    clf = DecisionTreeClassifier(max_depth=3).fit(X,y)
    fulltree = tree_to_code(clf, ["ft"+str(i) for i in range(X.shape[1])])
    # statistics
    linestats = "lines of full DT: {}, lines of interpretable DT: {}, simplification factor: {}"
    linestats = linestats.format(len(fulltree.split("\n")), len(itree.split("\n")), 1.0/len(fulltree.split("\n"))*len(itree.split("\n")))
    print("==== Interpretable DT for dataset `{}' ({}) ====".format(d,linestats))
    # print 
    print(itree)
    # add evaluation code
    ppred = []
    ftlist = ",".join(["ft"+str(i) for i in range(X.shape[1])])
    itree += "\nfor x in X:\n    "
    itree += ftlist + " = x\n"
    itree += "    ppred.append(probability_of_class_one(" + ftlist + "))\n"
    exec(itree)
    # evaluate
    ypred = 1*(np.array(ppred)>0.5)
    print("==== end of DT for dataset `{}'. F1 score: {} ====".format(d,f1_score(y,ypred)))
    print("")