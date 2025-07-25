import pickle
import pandas as pd
import pickle
import os
import numpy as np
import seaborn as sns
import matplotlib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
matplotlib.use('Agg')  # Set backend to Agg to prevent GUI window
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    mean_squared_error,
    r2_score,
    accuracy_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings("ignore")
artifacts_path=os.path.join(os.path.dirname(__file__),"artifacts")
plot_path=os.path.join(artifacts_path,"Plots")
def train_model(data_path, dependent_feature,rmse_prob,f1_prob,n_jobs=-1,n_iter=100,cv=3):
    os.makedirs(artifacts_path,exist_ok=True)
    print("Getting data from:",data_path)
    data=pd.read_csv(data_path)
    df=pd.DataFrame(data)
    # os.remove(data_path)
    df.head()
    print("Data loaded successfully")

    for i in df.columns:
        if df[i].dtype=="object":
            df[i] = df[i].fillna(df[i].mode()[0])
        else:
            df[i] = df[i].fillna(df[i].median())
            
    mild=False
    moderate=False
    majority=max(df[dependent_feature].value_counts())
    minority=min(df[dependent_feature].value_counts())
    IR = majority/minority
    # print(f"Imbalance Ratio: {IR}")
    if IR <= 3:
        # print("Mild Imbalance")
        mild=True
    elif 3 < IR <= 20:
        # print("Moderate Imbalance")
        moderate=True

    # feature selection
    # Replace < Dependent feture > and < Independent feature > with actual column names
    x=df.drop(columns=[dependent_feature])
    y=df[dependent_feature]
    # print("Independent Feature:", x)
    # print("Dependent Feature:", y)
    regressor=False
    classification=False
    print("Finding the type of problem...")
    if df[dependent_feature].dtype=="object":
        classification=True
        print("Classification Problem")
    else:
        regressor=True
        print("Regression Problem")

    from sklearn.model_selection import train_test_split
    # Splitting the dataset into training and testing sets
    print("Splitting the dataset into training and testing sets...")
    if classification:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42,stratify=y)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    print("Dataset split successfully")

    cat_features=[i for i in x_train.columns if x_train[i].dtype=="object" and i!=dependent_feature]
    num_features=[i for i in x_train.columns if x_train[i].dtype!="object" and i!=dependent_feature]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        x[num_features].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        annot_kws={"size": 14}
    )
    plt.title("Correlation Heatmap", fontsize=18)
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path,"correlation_heatmap.png"), bbox_inches='tight')

    plt.close()
    def correlation(dataset,threshold):
        corr_thresh=set()
        for i in range(len(dataset.columns)):
            for j in range(i):
                colname=dataset.columns[i]
                if abs(dataset.corr().iloc[i,j])>threshold :
                    corr_thresh.add(colname)
        return corr_thresh

    dropcorr = correlation(x_train[num_features], 0.85)
    x_train.drop([col for col in dropcorr if col in x_train.columns], axis=1, inplace=True)
    x_test.drop([col for col in dropcorr if col in x_test.columns], axis=1, inplace=True)
    cat_features = [i for i in x_train.columns if x_train[i].dtype == "object" and i != dependent_feature]
    num_features = [i for i in x_train.columns if x_train[i].dtype != "object" and i != dependent_feature]
    print("Categorical features : ",cat_features)
    print("Numerical features : ",num_features)
    print("Balancing the dataset...")
    if mild:
        from imblearn.under_sampling import RandomUnderSampler
        undersampler = RandomUnderSampler(random_state=42)
        x_train, y_train = undersampler.fit_resample(x_train, y_train)
        print("Mild Imbalance Resolved")
    if moderate:
        from imblearn.over_sampling import SMOTETomek
        smote_tomek = SMOTETomek(random_state=42)
        x_train, y_train = smote_tomek.fit_resample(x_train, y_train)
        print("Moderate Imbalace Resolved")


    
    print("Preprocessing the data...")  
    from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder,LabelEncoder
    from sklearn.compose import ColumnTransformer
    scaler=StandardScaler()
    ohe=OneHotEncoder(drop="first")
    oe=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ohe_data=[]
    oe_data=[]
    for i in cat_features:
        if len(x_train[i].unique())>10:
            oe_data.append(i)
        else:
            ohe_data.append(i)
    # print("One Hot Encoding Features:", ohe_data)
    # print("Ordinal Encoding Features:", oe_data)
    preprocessor=ColumnTransformer(
        [("OneHotEncoder",ohe,ohe_data),
        ("OrdinalEncoder",oe,oe_data),
        ("StandardScaler",scaler,num_features)]
    )
    feature_names=list(x_train.columns)
    x_train=preprocessor.fit_transform(x_train)
    x_test=preprocessor.transform(x_test)
    if classification:
        le=LabelEncoder()
        y_train=le.fit_transform(y_train)
        y_test=le.transform(y_test)
    model_dict=[]
    print("Training the models...")
    if regressor:
        from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
        from sklearn.linear_model import LinearRegression,Ridge, Lasso,ElasticNet
        from xgboost import XGBRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.svm import SVR
        from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
        models={
            "RandomForestRegressor":RandomForestRegressor(),
            "GradientBoostingRegressor":GradientBoostingRegressor(),
            "AdaBoostRegressor":AdaBoostRegressor(),
            "XGBRegressor":XGBRegressor(),
            "KNeighborsRegressor":KNeighborsRegressor(),
            "LinearRegression":LinearRegression(),
            "Ridge":Ridge(),
            "Lasso":Lasso(),
            "ElasticNet":ElasticNet(),
            "DecisionTreeRegressor":DecisionTreeRegressor(),
            "SVR":SVR()
        }
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train, y_train) # Train model

            # Make predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            model_train_mae = mean_absolute_error(y_train, y_train_pred)
            model_train_mse = mean_squared_error(y_train, y_train_pred)
            model_train_rmse = np.sqrt(model_train_mse)
            model_train_r2 = r2_score(y_train, y_train_pred)

            model_test_mae = mean_absolute_error(y_test, y_test_pred)
            model_test_mse = mean_squared_error(y_test, y_test_pred)
            model_test_rmse = np.sqrt(model_test_mse)
            model_test_r2 = r2_score(y_test, y_test_pred)
            model_dict.append({
            "model":list(models.keys())[i],
            "train_rmse": model_train_rmse,
            "train_r2": model_train_r2,
            "test_rmse": model_test_rmse,
            "test_r2": model_test_r2,
            "tuned":False
        })

            # print(list(models.keys())[i])
            
            # print('Model performance for Training set')
            # print("- mae: {:.4f}".format(model_train_mae))
            # print('-mse: {:.4f}'.format(model_train_mse))
            # print('- rmse: {:.4f}'.format(model_train_rmse))
            # print('- r2: {:.4f}'.format(model_train_r2))

            
            
            # print('----------------------------------')
            
            # print('Model performance for Test set')
            # print('- mae: {:.4f}'.format(model_test_mae))
            # print('- mse: {:.4f}'.format(model_test_mse))
            # print('- rmse: {:.4f}'.format(model_test_rmse))
            # print('- r2: {:.4f}'.format(model_test_r2))

            
            # print('='*35)
            # print('\n')
    else:
        from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
        from xgboost import XGBClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score,roc_auc_score,roc_curve,precision_score,recall_score
        models={
        "LogisticRegression":LogisticRegression(),
        "DecisionTreeClassifier":DecisionTreeClassifier(),
        "RandomForestClassifier":RandomForestClassifier(),
        "GradientBoostingClassifier":GradientBoostingClassifier(),
        "AdaBoostClassifier":AdaBoostClassifier(),
        "XGBClassifier":XGBClassifier(),
        "KNeighborsClassifier":KNeighborsClassifier(),
        "SVC":SVC()
    }
    
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train, y_train) # Train model

            # Make predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Training set performance
            model_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
            model_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score
            model_train_precision = precision_score(y_train, y_train_pred) # Calculate Precision
            model_train_recall = recall_score(y_train, y_train_pred) # Calculate Recall
            model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)
            # Test set performance
            model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
            model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score
            model_test_precision = precision_score(y_test, y_test_pred) # Calculate Precision
            model_test_recall = recall_score(y_test, y_test_pred) # Calculate Recal

            model_dict.append({
                "model":list(models.keys())[i],
                "train_accuracy": model_train_accuracy,
                "train_f1": model_train_f1,
                "test_accuracy": model_test_accuracy,
                "test_f1": model_test_f1,
                "tuned":False
            })
            model_test_rocauc_score = roc_auc_score(y_test, y_test_pred) #Calculate Roc


            # print(list(models.keys())[i])
            
            # print('Model performance for Training set')
            # print("- Accuracy: {:.4f}".format(model_train_accuracy))
            # print('- F1 score: {:.4f}'.format(model_train_f1))
            
            # print('- Precision: {:.4f}'.format(model_train_precision))
            # print('- Recall: {:.4f}'.format(model_train_recall))
            # print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))

            
            
            # print('----------------------------------')
            
            # print('Model performance for Test set')
            # print('- Accuracy: {:.4f}'.format(model_test_accuracy))
            # print('- F1 score: {:.4f}'.format(model_test_f1))
            # print('- Precision: {:.4f}'.format(model_test_precision))
            # print('- Recall: {:.4f}'.format(model_test_recall))
            # print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))

            
            # print('='*35)
            # print('\n')
    print("Finding the best model...")
    if regressor:
        comparison_df=pd.DataFrame(model_dict)
        comparison_df["total_rmse"]=comparison_df["train_rmse"]+comparison_df["test_rmse"]
        comparison_df["total_r2"]=comparison_df["train_r2"]+comparison_df["test_r2"]
        # print(comparison_df)
        comparison_df['Norm RMSE'] = comparison_df["total_rmse"] / comparison_df['total_rmse'].max()
        comparison_df['Norm R2'] = 1 - (comparison_df["total_r2"] / comparison_df['total_r2'].max())
        comparison_df['Combined Score'] = rmse_prob * comparison_df['Norm RMSE'] + (1-rmse_prob)* comparison_df['Norm R2']
        combined_score_ranking = comparison_df.sort_values('Combined Score')
        # print(combined_score_ranking[["train_rmse", "train_r2", "test_rmse", "test_r2","total_rmse","total_r2","Combined Score"]])
        # print("===="*35)
        best_models = comparison_df.nsmallest(3, 'Combined Score')

    if classification:
        comparison_df=pd.DataFrame(model_dict)
        comparison_df["total_accuracy"]=comparison_df["train_accuracy"]+comparison_df["test_accuracy"]
        comparison_df["total_f1"]=comparison_df["train_f1"]+comparison_df["test_f1"]
        # print(comparison_df)
        comparison_df['Norm F1'] = comparison_df["total_f1"] / comparison_df['total_f1'].max()
        comparison_df['Norm Accuracy'] = comparison_df["total_accuracy"] / comparison_df['total_accuracy'].max()
        comparison_df['Combined Score'] = f1_prob * comparison_df['Norm F1'] + (1-f1_prob)* comparison_df['Norm Accuracy']
        combined_score_ranking = comparison_df.sort_values('Combined Score',ascending=False)
        combined_score_ranking=combined_score_ranking[combined_score_ranking["train_f1"]-combined_score_ranking["test_f1"]<0.15]
        # print(combined_score_ranking[["model","train_accuracy", "train_f1", "test_accuracy", "test_f1","total_accuracy","total_f1","Combined Score"]])
        # print("===="*35)
        best_models = combined_score_ranking.nlargest(3, 'Combined Score')
        
    # print("Best Models based on Combined Score:", best_models[["model"]])
    # print("===="*35)
    # print(best_models)
    # print("===="*35)
    top_model=[i  for i in best_models["model"] ]
    print("Top Model:",top_model)
    print("Training the best models with enhanced parameters...")
    randomcv_model = []
    if regressor:
        reg_params = [
            ["RandomForestRegressor", {
                'n_estimators': [100, 200,300,400],
                'max_depth': [None, 5, 10, 15, 20, 30, 50],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 6],
                'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
                'bootstrap': [True, False],
                'random_state': [42]
            }],
            
            ["GradientBoostingRegressor", {
                'n_estimators': [100, 200,300,400],
                'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None],
                'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                'random_state': [42]
            }],
            
            ["AdaBoostRegressor", {
                'n_estimators': [50, 100, 150, 200, 300],
                'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                'loss': ['linear', 'square', 'exponential'],
                'random_state': [42]
            }],
            
            ["XGBRegressor", {
                'n_estimators': [100, 200,300,400],
                'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6, 7, 8, 9],
                'min_child_weight': [1, 2, 3, 4],
                'gamma': [0, 0.1, 0.2, 0.3, 0.4],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0, 0.1, 1, 10],
                'random_state': [42]
            }],
            
            ["KNeighborsRegressor", {
                'n_neighbors': [3, 5, 7, 10, 15, 20],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 20, 30, 40, 50],
                'p': [1, 2, 3]
            }],
            
            ["LinearRegression", {
                'fit_intercept': [True, False],
                'positive': [True, False],
                'copy_X': [True, False]
            }],
            
            ["Ridge", {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                'fit_intercept': [True, False],
                'random_state': [42]
            }],
            
            ["Lasso", {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'max_iter': [1000, 2000, 5000, 10000],
                'selection': ['cyclic', 'random'],
                'random_state': [42]
            }],
            
            ["ElasticNet", {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'max_iter': [1000, 5000],
                'selection': ['cyclic', 'random'],
                'random_state': [42]
            }],
            
            ["DecisionTreeRegressor", {
                'max_depth': [None, 5, 10, 15, 20, 30],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 6, 8],
                'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'random_state': [42]
            }],
            
            ["SVR", {
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                'C': [0.1, 0.5, 1, 5, 10, 50, 100],
                'epsilon': [0.01, 0.1, 0.2, 0.5, 1.0],
                'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10],
                'degree': [2, 3, 4]  # Only for poly kernel
            }]
        ]
        
        for i in reg_params:
            if i[0] in top_model:
                randomcv_model.append((i[0], models[i[0]], i[1]))
        
        # print("Enhanced Random CV Model Parameters:", randomcv_model)
    if classification:
        class_params = [
            ["LogisticRegression", {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': np.logspace(-4, 4, 20),
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [100, 200, 500, 1000],
                'class_weight': [None, 'balanced'],
                'l1_ratio': [0, 0.25, 0.5, 0.75, 1]  # For elasticnet
            }],
            
            ["DecisionTreeClassifier", {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'splitter': ['best', 'random'],
                'max_depth': [None, 5, 10, 15, 20, 30, 50],
                'min_samples_split': [2, 5, 10, 15, 20],
                'min_samples_leaf': [1, 2, 4, 6, 8],
                'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
                'class_weight': [None, 'balanced'],
                'random_state': [42]
            }],
            
            ["RandomForestClassifier", {
                'n_estimators': [100, 200,300,400],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': [None, 5, 10, 15, 20, 30, 50],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 6],
                'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced', 'balanced_subsample'],
                'random_state': [42]
            }],
            
            ["GradientBoostingClassifier", {
                'n_estimators': [100, 200,300,400],
                'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                'loss': ['log_loss', 'exponential'],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None],
                'random_state': [42]
            }],
            
            ["AdaBoostClassifier", {
                'n_estimators': [50, 100, 150, 200,300,400],
                'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                'random_state': [42]
            }],
            
            ["XGBClassifier", {
                'n_estimators': [100, 200,300 ,400],
                'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6, 7, 8, 9],
                'min_child_weight': [1, 2, 3, 4],
                'gamma': [0, 0.1, 0.2, 0.3, 0.4],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0, 0.1, 1, 10],
                'scale_pos_weight': [1, 5, 10],  # For imbalanced classes
                'random_state': [42],
                'use_label_encoder': [False],
                'eval_metric': ['logloss', 'aucpr', 'auc']
            }],
            
        
            
            ["KNeighborsClassifier", {
                'n_neighbors': [3, 5, 7, 10, 15, 20, 25],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 20, 30, 40, 50],
                'p': [1, 2, 3],
                'metric': ['minkowski', 'euclidean', 'manhattan']
            }],
            
            ["SVC", {
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'C': [0.001, 0.01, 0.1, 1, 5, 10, 50, 100],
                'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10],
                'degree': [2, 3, 4, 5],  # For poly kernel
                'class_weight': [None, 'balanced'],
                'probability': [True],  # If you need predict_proba
                'random_state': [42]
            }],
            
        
        ]
        
        for i in class_params:
            if i[0] in top_model:
                randomcv_model.append((i[0], models[i[0]], i[1]))
        
        # print("Enhanced Classification Model Parameters:", randomcv_model)

    from sklearn.model_selection import RandomizedSearchCV

    model_param = {}
    for name, model, params in randomcv_model:
        random = RandomizedSearchCV(estimator=model,
                                    param_distributions=params,
                                    n_iter=n_iter,
                                    cv=cv,
                                    n_jobs=n_jobs)
        random.fit(x_train, y_train)
        model_param[name] = random.best_params_

    # for model_name in model_param:
    #     print(f"---------------- Best Params for {model_name} -------------------")
    #     print(model_param[model_name])
    best_params=[]
    if regressor:
        model_cls={
                "RandomForestRegressor":RandomForestRegressor,
                "GradientBoostingRegressor":GradientBoostingRegressor,
                "AdaBoostRegressor":AdaBoostRegressor,
                "XGBRegressor":XGBRegressor,
                "KNeighborsRegressor":KNeighborsRegressor,
                "LinearRegression":LinearRegression,
                "Ridge":Ridge,
                "Lasso":Lasso,
                "ElasticNet":ElasticNet,
                "DecisionTreeRegressor":DecisionTreeRegressor,
                "SVR":SVR
            }
    if classification:
        model_cls={
                "LogisticRegression":LogisticRegression,
                "DecisionTreeClassifier":DecisionTreeClassifier,
                "RandomForestClassifier":RandomForestClassifier,
                "GradientBoostingClassifier":GradientBoostingClassifier,
                "AdaBoostClassifier":AdaBoostClassifier,
                "XGBClassifier":XGBClassifier,
                "KNeighborsClassifier":KNeighborsClassifier,
                "SVC":SVC
            }
    for i in model_param:
        best_params.append([i,model_param[i]])
    # print("Best Params:", best_params)

    best_models_copy=best_models.copy()
    if regressor:
        best_models_copy.drop(columns=['total_rmse'	,'total_r2'	,'Norm RMSE','Norm R2',"Combined Score"], inplace=True) 
    if classification:
        best_models_copy.drop(columns=['total_accuracy'	,'total_f1'	,'Norm F1','Norm Accuracy',"Combined Score"], inplace=True)
    models={}
    for i in best_params:
        models[i[0]]=model_cls[i[0]](**i[1])
    if regressor:
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train, y_train) # Train model

            # Make predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            model_train_mae = mean_absolute_error(y_train, y_train_pred)
            model_train_mse = mean_squared_error(y_train, y_train_pred)
            model_train_rmse = np.sqrt(model_train_mse)
            model_train_r2 = r2_score(y_train, y_train_pred)

            model_test_mae = mean_absolute_error(y_test, y_test_pred)
            model_test_mse = mean_squared_error(y_test, y_test_pred)
            model_test_rmse = np.sqrt(model_test_mse)
            model_test_r2 = r2_score(y_test, y_test_pred)

            # print(list(models.keys())[i])
            model_dict={
                "model":list(models.keys())[i],
                "train_rmse": model_train_rmse,
                        "train_r2": model_train_r2,
                        "test_rmse": model_test_rmse,
                        "test_r2": model_test_r2,
                        "tuned":True}
            best_models_copy = pd.concat(
        [best_models_copy, pd.DataFrame([model_dict])],
        ignore_index=True
    )

            
            # print('Model performance for Training set')
            # print("- mae: {:.4f}".format(model_train_mae))
            # print('-mse: {:.4f}'.format(model_train_mse))
            # print('- rmse: {:.4f}'.format(model_train_rmse))
            # print('- r2: {:.4f}'.format(model_train_r2))

            
            
            # print('----------------------------------')
            
            # print('Model performance for Test set')
            # print('- mae: {:.4f}'.format(model_test_mae))
            # print('- mse: {:.4f}'.format(model_test_mse))
            # print('- rmse: {:.4f}'.format(model_test_rmse))
            # print('- r2: {:.4f}'.format(model_test_r2))

            
            # print('='*35)
            # print('\n')

    if classification:
    
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train, y_train) # Train model

            # Make predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Training set performance
            model_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
            model_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score
            model_train_precision = precision_score(y_train, y_train_pred) # Calculate Precision
            model_train_recall = recall_score(y_train, y_train_pred) # Calculate Recall
            model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)
            # Test set performance
            model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
            model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score
            model_test_precision = precision_score(y_test, y_test_pred) # Calculate Precision
            model_test_recall = recall_score(y_test, y_test_pred) # Calculate Recal

            model_dict={
            "model":list(models.keys())[i],
            "train_accuracy": model_train_accuracy,
            "train_f1": model_train_f1,
            "test_accuracy": model_test_accuracy,
            "test_f1": model_test_f1,
            "tuned":True
        }
            best_models_copy = pd.concat(
        [best_models_copy, pd.DataFrame([model_dict])],
        ignore_index=True
    )
            


            # print(list(models.keys())[i])
            
            # print('Model performance for Training set')
            # print("- Accuracy: {:.4f}".format(model_train_accuracy))
            # print('- F1 score: {:.4f}'.format(model_train_f1))
            
            # print('- Precision: {:.4f}'.format(model_train_precision))
            # print('- Recall: {:.4f}'.format(model_train_recall))
            # print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))

            
            
            # print('----------------------------------')
            
            # print('Model performance for Test set')
            # print('- Accuracy: {:.4f}'.format(model_test_accuracy))
            # print('- F1 score: {:.4f}'.format(model_test_f1))
            # print('- Precision: {:.4f}'.format(model_test_precision))
            # print('- Recall: {:.4f}'.format(model_test_recall))
            # print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))

            
            # print('='*35)
            # print('\n')
    best_models_copy
    if regressor:
        best_models_copy["total_rmse"]=best_models_copy["train_rmse"]+best_models_copy["test_rmse"]
        best_models_copy["total_r2"]=best_models_copy["train_r2"]+best_models_copy["test_r2"]
        best_models_copy['Norm RMSE'] = best_models_copy["total_rmse"] / best_models_copy['total_rmse'].max()
        best_models_copy['Norm R2'] = 1 - (best_models_copy["total_r2"] / best_models_copy['total_r2'].max())
        best_models_copy['Combined Score'] = rmse_prob * best_models_copy['Norm RMSE'] + (1-rmse_prob)* best_models_copy['Norm R2']
        combined_score_ranking = best_models_copy.sort_values('Combined Score').reset_index(drop=True)
    if classification:
        best_models_copy["total_accuracy"]=best_models_copy["train_accuracy"]+best_models_copy["test_accuracy"]
        best_models_copy["total_f1"]=best_models_copy["train_f1"]+best_models_copy["test_f1"]
        best_models_copy['Norm F1'] = best_models_copy["total_f1"] / best_models_copy['total_f1'].max()
        best_models_copy['Norm Accuracy'] = best_models_copy["total_accuracy"] / best_models_copy['total_accuracy'].max()
        best_models_copy['Combined Score'] = f1_prob * best_models_copy['Norm F1'] + (1-f1_prob)* best_models_copy['Norm Accuracy']
        combined_score_ranking = best_models_copy.sort_values('Combined Score',ascending=False).reset_index(drop=True)
    # print(combined_score_ranking.iloc[0:1,:]["model"][0])
    # print(combined_score_ranking)
    if classification:
        combined_score_ranking=combined_score_ranking[combined_score_ranking["train_f1"]-combined_score_ranking["test_f1"]<0.15]
    # print("====="*35)
    print(combined_score_ranking.iloc[0:1,:])
    best_model_name = combined_score_ranking.iloc[0]["model"]
    best_param_dict = None
    for name, params in best_params:
        if name == best_model_name:
            best_param_dict = params
            break
    model = model_cls[best_model_name](**best_param_dict)
    # print("Best Model Name:", best_model_name)
    # print("Best Model Parameters:", best_param_dict)
    model.fit(x_train, y_train)
    print("Best Model Trained Successfully")
    print("Saving the model and preprocessor...")
    model_path = os.path.join(artifacts_path, "model.pkl")
    preprocessor_path = os.path.join(artifacts_path, "preprocessor.pkl")
    encoder_path = os.path.join(artifacts_path, "encoder.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)

    if classification:
        with open(encoder_path, "wb") as f:
            pickle.dump(le, f)
        response = {
        "message": "Training completed successfully",
        "Problem_type":"Classification",
        "model_metrics": {
            "Model": combined_score_ranking.iloc[0]["model"],
            "Train_accuracy": float(combined_score_ranking.iloc[0]["train_accuracy"]),
            "Train_f1": float(combined_score_ranking.iloc[0]["train_f1"]),
            "Test_accuracy": float(combined_score_ranking.iloc[0]["test_accuracy"]),
            "Test_f1": float(combined_score_ranking.iloc[0]["test_f1"]),
            "Hyper_tuned": bool(combined_score_ranking.iloc[0]["tuned"]),
            "Dropped_Columns":list(dropcorr)
        }
    }
        plot_classification_metrics(model,x_train, y_train, x_test, y_test, class_names=['No', 'Yes'])
    else:
        response={
             "message": "Training completed successfully",
             "Problem_type":"Regression",
            "model_metrics": {
            "Model": combined_score_ranking.iloc[0]["model"],
            "Train_R2": float(combined_score_ranking.iloc[0]["train_r2"]),
            "Train_RMSE": float(combined_score_ranking.iloc[0]["train_rmse"]),
            "Test_R2": float(combined_score_ranking.iloc[0]["test_r2"]),
            "Test_RMSE": float(combined_score_ranking.iloc[0]["test_rmse"]),
            "Hyper_tuned": bool(combined_score_ranking.iloc[0]["tuned"]),
            "Dropped_Columns":list(dropcorr)
            }
        }
        plot_regression_metrics(model, x_train, y_train, x_test, y_test,feature_names)
    print(response)
    return {"status": "success", "model": "trained_model.pkl"}

def plot_classification_metrics(model, X_train, y_train, X_test, y_test, class_names=None):
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plot_path,"confusion_matrix.png"), bbox_inches='tight')
    plt.close()
    # ROC Curve
    if y_proba is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.title("ROC Curve")
        plt.savefig(os.path.join(plot_path,"roc_curve.png"), bbox_inches='tight')
        plt.close()
    # Precision-Recall Curve
    if y_proba is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        PrecisionRecallDisplay(precision=precision, recall=recall).plot()
        plt.title("Precision-Recall Curve")
        plt.savefig(os.path.join(plot_path,"precision_recall_curve.png"), bbox_inches='tight')
        plt.close()
    # Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=5, scoring='accuracy')
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label="Validation")
    plt.title("Learning Curve (Accuracy)")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(plot_path,"Accuracy_curve.png"), bbox_inches='tight')
    plt.close()

    # Class Distribution
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(x=y_train, ax=ax[0])
    ax[0].set_title("Training Class Distribution")
    sns.countplot(x=y_test, ax=ax[1])
    ax[1].set_title("Testing Class Distribution")
    plt.savefig(os.path.join(plot_path,"class_distribution.png"), bbox_inches='tight')
    plt.close()
def plot_regression_metrics(model, X_train, y_train, X_test, y_test,feature_names):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Actual vs. Predicted
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs. Predicted")
    plt.savefig(os.path.join(plot_path,"actual_vs_predictes.png"), bbox_inches='tight')
    plt.close()
    # Residual Plot
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig(os.path.join(plot_path,"residual.png"), bbox_inches='tight')
    plt.close()
    # Distribution of Residuals
    sns.histplot(residuals, kde=True)
    plt.title("Distribution of Residuals")
    plt.xlabel("Residual")
    plt.savefig(os.path.join(plot_path,"residual_distribution.png"), bbox_inches='tight')
    plt.close()
    # Learning Curve (R² or MSE)
    train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=5, scoring='r2')
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label="Validation")
    plt.title("Learning Curve (R² Score)")
    plt.xlabel("Training Set Size")
    plt.ylabel("R² Score")
    plt.savefig(os.path.join(plot_path,"r2_score.png"), bbox_inches='tight')
    plt.legend()
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--rmse", required=True)
    parser.add_argument("--f1", required=True)
    parser.add_argument("--n_jobs", default=-1, type=int)
    parser.add_argument("--n_iter", default=100, type=int)
    parser.add_argument("--cv", default=3, type=int)
    args = parser.parse_args()
    print(train_model(args.data, args.target, rmse_prob=args.rmse, f1_prob=args.f1, n_jobs=args.n_jobs, n_iter=args.n_iter, cv=args.cv))

