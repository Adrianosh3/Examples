import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
# Cummulative ML Project Dating a Scientist

# df = dataframe
df = pd.read_csv("profiles.csv")
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

#print(df.status.head())
print(df.columns)

plt.hist(df.age, bins=30)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 75)
#plt.show()
print(df.job.value_counts())
# mapping
body_mapping = { "athletic": 2, "thin": 1, "a little extra": 0, "full figured": 0, "overweight": 0,
"jacked": 2, "used up": 1, "rather not say": 0}

drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}

smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "trying to quit": 3, "yes": 4}

drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}

sign_mapping = {"aquarius": 1,
                "aquarius and it matters a lot": 1,
   "aquarius and it&rsquo;s fun to think about": 1,
         "aquarius but it doesn&rsquo;t matter": 1,
                                        "aries": 2,
                   "aries and it matters a lot": 2,
      "aries and it&rsquo;s fun to think about": 2,
            "aries but it doesn&rsquo;t matter": 2,
                                       "cancer":3,
                  "cancer and it matters a lot":3,
     "cancer and it&rsquo;s fun to think about":3,
           "cancer but it doesn&rsquo;t matter":3,
                                    "capricorn":4,
               "capricorn and it matters a lot":4,
  "capricorn and it&rsquo;s fun to think about":4,
        "capricorn but it doesn&rsquo;t matter":4,
                                       "gemini":5,
                  "gemini and it matters a lot":5,
     "gemini and it&rsquo;s fun to think about":5,
           "gemini but it doesn&rsquo;t matter":5,
                                          "leo":6,
                     "leo and it matters a lot":6,
        "leo and it&rsquo;s fun to think about":6,
              "leo but it doesn&rsquo;t matter":6,
                                        "libra":7,
                   "libra and it matters a lot":7,
      "libra and it&rsquo;s fun to think about":7,
            "libra but it doesn&rsquo;t matter":7,
                                       "pisces":8,
                  "pisces and it matters a lot":8,
     "pisces and it&rsquo;s fun to think about":8,
           "pisces but it doesn&rsquo;t matter":8,
                                  "sagittarius":9,
             "sagittarius and it matters a lot":9,
"sagittarius and it&rsquo;s fun to think about":9,
      "sagittarius but it doesn&rsquo;t matter":9,
                                      "scorpio":10,
                 "scorpio and it matters a lot":10,
    "scorpio and it&rsquo;s fun to think about":10,
          "scorpio but it doesn&rsquo;t matter":10,
                                       "taurus":11,
                  "taurus and it matters a lot":11,
     "taurus and it&rsquo;s fun to think about":11,
           "taurus but it doesn&rsquo;t matter":11,
                                        "virgo":12,
                   "virgo and it matters a lot":12,
      "virgo and it&rsquo;s fun to think about":12,
            "virgo but it doesn&rsquo;t matter":12}

# highschool = 0, 2 year college = 1, undergrad = 2, masters = 3, phd = 4, med =5, law = 6, (working or dropouts or other)= NA
education_mapping = {"graduated from college/university": 2,
                "graduated from masters program": 3,
                "working on college/university": np.nan,
                "working on masters program": np.nan,
                "graduated from two-year college": 1,
                "graduated from high school": 0,
                "graduated from ph.d program": 4,
                "graduated from law school": 6,
                "working on two-year college": np.nan,
                "dropped out of college/university": np.nan,
                "working on ph.d program": np.nan,
                "college/university": 2,
                "graduated from space camp":  np.nan,
                "dropped out of space camp":  np.nan,
                "graduated from med school":  5,
                "working on space camp":  np.nan,
                "working on law school":  np.nan ,
                "two-year college":  1,
                "working on med school ":   np.nan,
                "dropped out of two-year college":   np.nan,
                "dropped out of masters program ":   np.nan,
                "masters program":   3,
                "dropped out of ph.d program":  np.nan,
                "dropped out of high school":  np.nan,
                "high school":   0,
                "working on high school":    np.nan,
                "space camp":     np.nan,
                "ph.d program":    4,
                "law school":      6,
                "dropped out of law school":    np.nan,
                "dropped out of med school":    np.nan,
                "med school":    5}

#relgion diets at 0, vegan at 1, vegetarian at 2, anything at 3, and other at 4

diet_mapping = {'halal': 0,
                'strictly halal': 0,
                'mostly halal': 0,
                'kosher': 0,
                'strictly kosher': 0,
                'mostly kosher': 0,
                'vegan': 1,
                'strictly vegan': 1,
                'mostly vegan': 1,
                'vegetarian': 2,
                'strictly vegetarian': 2,
                'mostly vegetarian': 2,
                'anything': 3,
                'strictly anything': 3,
                'mostly anything': 3,
                'other': 4,
                'strictly other': 4,
                'mostly other': 4}

df["signs_simple"] = df["sign"].map(sign_mapping)
df["smokes_code"] = df.smokes.map(smokes_mapping)
df["drugs_code"] = df.drugs.map(drugs_mapping)
df["drinks_code"] = df.drinks.map(drink_mapping)
df["education"] = df["education"].map(education_mapping)
df["diet"] = df["diet"].map(diet_mapping)
df["body_type"] = df["body_type"].map(body_mapping)
df["religion"] = df["religion"].str.split().str.get(0)


# get numerical data from text inputs
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

df["essay_len"] = all_essays.apply(lambda x: len(x))
df["avg_word_length"] = all_essays.apply(lambda x: (len(x) + 1)/(len(x.split())+ 1))

df = df.dropna(subset = ['signs_simple', 'smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'avg_word_length', "diet", "education", "religion", "body_type"])
def check_nan():
    for column in ['sign', 'smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'avg_word_length', "diet", "education", "religion", "body_type"]:
        print(column)
        print(df[column].unique())
        print(df[column].isna().any())
#print(check_nan())

new_columns = ['sign','smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'avg_word_length', "diet", "education", "religion", "body_type"]
new_df = df[new_columns]
cols = list(new_df.columns)
for col in cols[:-1]:
    new_df = pd.get_dummies(new_df, columns=[col], prefix = [col])
#print(new_df.head())


features = new_df.iloc[:, 1:len(new_df.columns)]
target = df["sex"]
print(new_df.shape)
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()

#test parameters
"""scores = []
for i in range(1,201):
  tree = DecisionTreeClassifier(random_state=1, max_depth = i)
  tree.fit(x_train, y_train)
  scores.append(tree.score(x_test,y_test))
plt.plot(range(1,201), scores) """

# Classfication techniques
# Decision Tree
tree_model = DecisionTreeClassifier(random_state=34, max_depth=6)
tree_model.fit(x_train, y_train)
print("DT train score:", tree_model.score(x_train, y_train))
print("DT test Score :", tree_model.score(x_test, y_test))
tree_predictions = tree_model.predict(x_train)
print(classification_report(y_train, tree_predictions))

# Random Forest
forest_model = RandomForestClassifier(random_state=34)
forest_model.fit(x_train, y_train)
forest_prediction = forest_model.predict(x_train)
print("RF training score :", forest_model.score(x_train, y_train))
print("RF test score: ", forest_model.score(x_test, y_test))
print(classification_report(y_train, forest_prediction))

#Normalizing data for these classifiers!
x = features.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

features_data = pd.DataFrame(x_scaled, columns=features.columns)
target_data = df["sex"]

X_train, X_test, Y_train, Y_test = train_test_split(features_data, target_data, test_size = 0.2, random_state = 42)
Y_train = Y_train.to_numpy().ravel()
Y_test = Y_test.to_numpy().ravel()



'''
accuracies = []
for k in range(1,101):
  classifer = KNeighborsClassifier(n_neighbors = k)
  classifer.fit(X_train, Y_train)
  k_list = range(1,101)
  accuracies.append(classifer.score(X_test, Y_test))
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("body_type Accuracy")
plt.show()
#results showed 20 was the best option
'''

classifier = KNeighborsClassifier(n_neighbors = 20)
classifier.fit(X_train, Y_train)
print("KNN train score", classifier.score(X_train, Y_train))
print("KNN test score", classifier.score(X_test, Y_test))
KNC_predict = classifier.predict(X_train)
print(classification_report(Y_train, KNC_predict))



model = LogisticRegression(multi_class = "multinomial", max_iter = 500)
model.fit(X_train, Y_train)
LR_predict = model.predict(X_train)
print("Predict prob:", model.predict_proba(X_train))
print("LR train score:", model.score(X_train,Y_train))
print("LR test score:", model.score(X_test,Y_test))
print(classification_report(Y_train, LR_predict))

'''
#Create a dictionary of possible parameters
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid', "linear"]}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,Y_train)
print(grid.best_estimator_)
#Results indicate that SVC(C=100, gamma=0.01) was the best option
'''


#vector naive bayes (SVM)
SVC_model = SVC(kernel = "rbf", gamma = 0.01, C = 100)
SVC_model.fit(X_train, Y_train)
SVC_predict = SVC_model.predict(X_train)
print("SVC train score:", SVC_model.score(X_train, Y_train))
print("SVC test score:", SVC_model.score(X_test, Y_test))
print(classification_report(Y_train, SVC_predict))
