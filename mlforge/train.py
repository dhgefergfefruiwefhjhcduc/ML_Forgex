from json import encoder
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
import pickle
import pandas as pd
import os
import tempfile
import shutil
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from mlforge.dashboard import generate_dashboard

matplotlib.use('Agg')  # Set backend to Agg to prevent GUI window
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import gensim
stopword=stopwords.words('english')
stopword.remove("not")
stopword.remove("nor")
stopword.remove("no")
lemmatizer=WordNetLemmatizer()


def train_model(data_path, dependent_feature,rmse_prob=0.7,f1_prob=0.3,n_jobs=-1,n_iter=100,n_splits=5,test_size=0.2,artifacts_dir=None,artifacts_name="artifacts",fast=False,corr_threshold=0.85,skew_threshold=1,z_threshold=3,overfit_threshold=0.15,nlp=False,dashboard_title="mlforgex Dashboard"):
    """
Trains and evaluates a machine learning model using the provided dataset, 
with automated preprocessing, feature selection, hyperparameter tuning, 
and overfitting detection.

Args:
    data_path (str):
        Path to the CSV dataset file. The dataset must contain the dependent
        variable (target) along with all the input features.

    dependent_feature (str):
        The name of the target column (dependent variable) in the dataset.
        This is the value the model will learn to predict.

    rmse_prob (float):
        Probability threshold for RMSE evaluation (used in regression problems).
        Helps in deciding acceptable error levels for regression models.

    f1_prob (float):
        Probability threshold for F1-score evaluation (used in classification problems).
        Helps in assessing classification performance, especially for imbalanced datasets.

    n_jobs (int, optional):
        Number of parallel jobs to run during model training and evaluation.
        -1 means using all available CPUs. Default is -1.

    n_iter (int, optional):
        Number of iterations for randomized hyperparameter search. 
        Higher values increase accuracy but require more computation time.
        Default is 100.

    n_splits (int, optional):
        Number of folds for cross-validation. 
        Determines how the dataset is split during validation.
        Default is 5.
    
    test_size (float, optional):
        Proportion of the dataset to include in the test split.
        Must be between 0.0 and 1.0. Default is 0.2.

    fast (bool, optional):
        If True, uses a faster but less exhaustive hyperparameter tuning approach
        for quicker results. If False, performs a more thorough search. 
        Default is False.

    artifacts_dir (str, optional):
        Directory to save training artifacts such as models, plots, and logs.
        If None, the current working directory is used.
        Default is None.

    artifacts_name (str, optional):
        Name of the artifacts directory (inside artifacts_dir).
        Useful for organizing multiple training runs.
        Default is "artifacts".

    corr_threshold (float, optional):
        Maximum correlation threshold for feature selection.
        Features with correlations higher than this value (highly collinear features) may be dropped.
        Default is 0.85.

    skew_threshold (float, optional):
        Threshold for handling skewness in numerical features.
        Features with skewness beyond this value may be transformed to reduce skewness.
        Default is 1.

    z_threshold (float, optional):
        Z-score threshold for outlier removal. 
        Data points with Z-scores beyond this threshold are considered outliers and may be removed.
        Default is 3.

    overfit_threshold (float, optional):
        Specifies the maximum acceptable gap between the training F1-score 
        and the testing F1-score. If the difference between them exceeds this threshold,
        the model is flagged as overfitting.
        - A smaller value (e.g., 0.05) makes overfitting detection stricter.
        - A larger value (e.g., 0.3) allows more tolerance.
        Default is 0.15.
    
    nlp (bool, optional):
        If True, enable NLP/text-mode: combine text columns (or use 'text'), run tokenization,
        stopword removal and lemmatization, vectorize text (Word2Vec by default), enforce label
        encoding and classification flow, and save the Word2Vec model to artifacts. Default False.

    dashboard_title (str, optional):
        Title for the dashboard generated after training. Default is "mlforgex Dashboard".

Returns:
    dict:
        A dictionary containing:
            - Model evaluation metrics (e.g., accuracy, F1-score, RMSE).
            - The trained model object.
    """

    if artifacts_dir:
        artifacts_path = os.path.join(artifacts_dir, artifacts_name)
    else:
        artifacts_path = os.path.join(os.getcwd(), artifacts_name)
    temp_path=os.path.join(artifacts_path, "temp")
    os.makedirs(temp_path, exist_ok=True) 
    tempfile.tempdir= temp_path
    os.makedirs(artifacts_path, exist_ok=True)
    print("Getting data from:", data_path)
    data=pd.read_csv(data_path)
    df=pd.DataFrame(data)
    # os.remove(data_path)
    print("Data loaded successfully")
    regressor=False
    classification=False
    encode=False
    mild=False
    moderate=False
    corr_thresh=set()
    dropcorr=set()
    cat_features=[]
    num_features=[]
    plots=[]
    if nlp:
        text_col=[i for i in df.columns if df[i].dtype=="object" and i!=dependent_feature]
        cat_col = [i for i in df.columns if df[i].dtype == "object" and i != dependent_feature 
           and (df[i].str.split().str.len() == 1).all()]
        for col in cat_col:
            mode_val = df[col].mode()
            fill_value = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].apply(
                lambda x: fill_value if pd.isna(x) or str(x).lower().strip() in ["nan", "null", "none", "nil", "na"] else str(x)
            )
        dropcorr.update([i for i in df.columns if i not in text_col and i!=dependent_feature])
        cat_features=text_col.copy()
        if "text" in df.columns:
            df["text"] = df["text"].apply(lambda x: "" if pd.isna(x) or str(x).lower().strip() in ["nan", "null", "none", "nil", "na"] else str(x))
            other_text_cols = [c for c in text_col if c != "text"]
            for col in other_text_cols:
                df[col] = df[col].apply(lambda x: "" if pd.isna(x) or str(x).lower().strip() in ["nan", "null", "none", "nil", "na"] else str(x))
            if other_text_cols:
                df["text"] = df["text"].fillna("").astype(str) + " " + df[other_text_cols].astype(str).agg(" ".join, axis=1)
                df["text"] = df["text"].str.strip()
                df.drop(columns=other_text_cols, inplace=True)
        else:
            if text_col:
                for col in text_col:
                    df[col] = df[col].apply(lambda x: "" if pd.isna(x) or str(x).lower().strip() in ["nan", "null", "none", "nil", "na"] else str(x))
                df["text"] = df[text_col].astype(str).agg(" ".join, axis=1).str.strip()
                df.drop(columns=text_col, inplace=True)
            else:
                df["text"] = "" 
        from mlforge.cleaning import preprocess
        df.drop(columns=dropcorr,inplace=True)
        x=df[["text"]]
        y=df[dependent_feature]
        is_multiclass = len(set(y)) > 2
        average_type = "weighted" if is_multiclass else "binary"
        print("preprocessing text...")
        x.loc[:, "text"] = x["text"].apply(preprocess)
        mask = x["text"].str.strip() != ""
        x = x[mask]
        y = y[mask]
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        if df[dependent_feature].dtype=="object": encode=True
        
    else:
        from mlforge.cleaning import data_cleaning
        df=data_cleaning(df,skew_threshold,z_threshold,dependent_feature)
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
        dropcorr.update([i for i in df.columns if df[i].nunique() == 1 or df[i].isna().sum()/len(df)>=0.6])
        df.drop(columns=dropcorr, axis=1,inplace=True)
        dropcorr.update([i for i in df.columns if df[i].nunique()==df.shape[0] and df[i].dtype in ["int64","object"]])
        df.drop(columns=[col for col in df.columns if df[col].nunique()==df.shape[0] and df[col].dtype in ["int64","object"]], inplace=True)
        x=df.drop(columns=[dependent_feature])
        y=df[dependent_feature]
        # print("Independent Feature:", x)
        # print("Dependent Feature:", y)
        print("Finding the type of problem...")
        if df[dependent_feature].dtype=="object":
            classification=True
            encode=True
            print("Classification Problem")
        else:
            if(df[dependent_feature].nunique() < 20):
                classification=True
                print("Classification Problem")
            else:
                regressor=True
                print("Regression Problem")
        corr_thresh.update([i for i in df.columns if df[i].nunique() == 1])
    from sklearn.model_selection import train_test_split
    # Splitting the dataset into training and testing sets
    print("Splitting the dataset into training and testing sets...")
    if classification or nlp:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42,stratify=y)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    print("Dataset split successfully")
    if nlp:
        from mlforge.cleaning import avg_wordtovec
        print("Generating word vectors...")
        word_token_train=[word_tokenize(i) for i in x_train["text"]]
        mod=gensim.models.Word2Vec(word_token_train)
        mod.save(os.path.join(artifacts_path,"word2vec.model"))
        x_train=[avg_wordtovec(i,mod) for i in word_token_train]
        word_token_test=[word_tokenize(i) for i in x_test["text"]]
        x_test=[avg_wordtovec(i,mod) for i in word_token_test]
    if not nlp:
        cat_features=[i for i in x_train.columns if x_train[i].dtype=="object" and i!=dependent_feature]
        num_features=[i for i in x_train.columns if x_train[i].dtype!="object" and i!=dependent_feature]
        
        if len(num_features) > 0:
                    if num_features is None:
                        num_features = x.select_dtypes(include=[np.number]).columns.tolist()
                    
                    # Calculate correlation matrix
                    corr_matrix = x[num_features].corr()
                    
                    # Create Plotly heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='RdBu_r',  # Red-Blue reversed (similar to coolwarm)
                        zmin=-1,  # Fixed range for correlation
                        zmax=1,
                        hoverongaps=False,
                        hovertemplate=(
                            '<b>X:</b> %{x}<br>' +
                            '<b>Y:</b> %{y}<br>' +
                            '<b>Correlation:</b> %{z:.3f}<br>' +
                            '<extra></extra>'
                        ),
                        colorbar=dict(
                            title="Correlation",
                            title_side="right",
                            tickvals=[-1, -0.5, 0, 0.5, 1],
                            ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
                        )
                    ))
                    
                    # Add annotations (correlation values)
                    for i in range(len(corr_matrix.columns)):
                        for j in range(len(corr_matrix.index)):
                            fig.add_annotation(
                                x=corr_matrix.columns[i],
                                y=corr_matrix.index[j],
                                text=f"{corr_matrix.iloc[j, i]:.2f}",
                                showarrow=False,
                                font=dict(
                                    size=12,
                                    color='white' if abs(corr_matrix.iloc[j, i]) > 0.5 else 'black'
                                )
                            )
                    
                    # Update layout
                    fig.update_layout(
                        title=dict(
                            text="Correlation Heatmap",
                            x=0.5,
                            font=dict(size=24, color='black')
                        ),
                        xaxis=dict(
                            title="Features",
                            tickangle=45,
                            tickfont=dict(size=12),
                            side="bottom"
                        ),
                        yaxis=dict(
                            title="Features", 
                            tickfont=dict(size=12),
                            autorange="reversed"  # Match seaborn orientation
                        ),
                        template='plotly_white',
                        width=1000,
                        height=800,
                        margin=dict(l=50, r=50, t=100, b=100)
                    )
                    
                    plots.append(("Correlation Heatmap", fig))
   
        def correlation(df,dataset,target,max_threshold):
            dataset=dataset.copy()
            corr_matrix=dataset.corr()
            for i in range(len(dataset.columns)):
                for j in range(i):
                    colname = corr_matrix.columns[i]
                    corr_value = abs(corr_matrix.iloc[i, j])
                    if corr_value > max_threshold :
                        corr_thresh.add(colname)

            return corr_thresh
        dropcorr.update(correlation(df,x_train[num_features],dependent_feature,corr_threshold))
        x_train.drop([col for col in corr_thresh if col in x_train.columns], axis=1, inplace=True)
        x_test.drop([col for col in corr_thresh if col in x_test.columns], axis=1, inplace=True)
        cat_features = [i for i in x_train.columns if x_train[i].dtype == "object" and i != dependent_feature]
        num_features = [i for i in x_train.columns if x_train[i].dtype != "object" and i != dependent_feature]
    if classification or nlp:
        is_multiclass = len(set(y_train)) > 2
        average_type = "weighted" if is_multiclass else "binary"
    ohe_data=[]
    oe_data=[]
    if not nlp:   
        print("Preprocessing the data...")  
        from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder,LabelEncoder
        from sklearn.compose import ColumnTransformer
        scaler=StandardScaler()
        ohe=OneHotEncoder(drop="first",handle_unknown="ignore")
        oe=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
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
        x_train=preprocessor.fit_transform(x_train)
        x_test=preprocessor.transform(x_test)
        feature_names=preprocessor.get_feature_names_out()
    if classification or nlp:
        print("Balancing the dataset...")
        if mild :
            from imblearn.under_sampling import RandomUnderSampler
            undersampler = RandomUnderSampler(random_state=42)
            x_train, y_train = undersampler.fit_resample(x_train, y_train)
            print("Mild Imbalance Resolved")
        if moderate:
            from imblearn.combine import SMOTETomek
            smote_tomek = SMOTETomek(random_state=42)
            x_train, y_train = smote_tomek.fit_resample(x_train, y_train)
            print("Moderate Imbalace Resolved")
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    if (classification and encode) or nlp:
        y_train=le.fit_transform(y_train)
        y_test=le.transform(y_test)
        encode=True
    model_dict=[]
    # print("Training the models...")
    if regressor:
        from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
        from sklearn.linear_model import LinearRegression,Ridge, Lasso,ElasticNet
        from xgboost import XGBRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.svm import SVR
        from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
        models={
            "RandomForestRegressor":RandomForestRegressor(n_jobs=n_jobs),
            "GradientBoostingRegressor":GradientBoostingRegressor(),
            "AdaBoostRegressor":AdaBoostRegressor(),
            "XGBRegressor":XGBRegressor(),
            "KNeighborsRegressor":KNeighborsRegressor(n_jobs=n_jobs),
            "LinearRegression":LinearRegression(n_jobs=n_jobs),
            "Ridge":Ridge(),
            "Lasso":Lasso(),
            "ElasticNet":ElasticNet(),
            "DecisionTreeRegressor":DecisionTreeRegressor(),
            "SVR":SVR()
        }
        
        for i in tqdm(range(len(list(models))),desc="Training Models",unit="model"):
            model = list(models.values())[i]
            # print("-> "+list(models.keys())[i],flush=True)
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
            "train_mae": model_train_mae,
            "test_mae": model_test_mae,
            "train_mse": model_train_mse,
            "test_mse": model_test_mse,
            "tuned":False
        })

    else:
        from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
        from xgboost import XGBClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC,LinearSVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression,SGDClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score,roc_auc_score,roc_curve,precision_score,recall_score
        models={
        "LogisticRegression":LogisticRegression(n_jobs=n_jobs,max_iter=3000),
        "DecisionTreeClassifier":DecisionTreeClassifier(),
        "RandomForestClassifier":RandomForestClassifier(n_jobs=n_jobs),
        "GradientBoostingClassifier":GradientBoostingClassifier(),
        "AdaBoostClassifier":AdaBoostClassifier(),
        "XGBClassifier":XGBClassifier(n_jobs=n_jobs),
        "KNeighborsClassifier":KNeighborsClassifier(n_jobs=n_jobs),
        "SGDClassifier":SGDClassifier(n_jobs=n_jobs),
    }
        if nlp:
            models={
                "GaussianNB":GaussianNB(),
                "LogisticRegression":LogisticRegression(n_jobs=n_jobs,max_iter=3000),
                "LinearSVC":LinearSVC(),
                "RandomForestClassifier":RandomForestClassifier(n_jobs=n_jobs),
                "XGBClassifier":XGBClassifier(n_jobs=n_jobs),
            }
        for i in tqdm(range(len(list(models))), desc="Training Models", unit="model"):
            model = list(models.values())[i]
            # print("-> "+list(models.keys())[i],flush=True)
            model.fit(x_train, y_train) # Train model

            # Make predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Training set performance
            model_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
            model_train_f1 = f1_score(y_train, y_train_pred, average=average_type) # Calculate F1-score
            model_train_precision = precision_score(y_train, y_train_pred, average=average_type,zero_division=0) # Calculate Precision
            model_train_recall = recall_score(y_train, y_train_pred, average=average_type,zero_division=0) # Calculate Recall
            if hasattr(model,"predict_proba"):
                if average_type == "binary":
                    y_train_prob = model.predict_proba(x_train)[:,1]
                    model_train_rocauc_score = roc_auc_score(y_train, y_train_prob)
                else:
                    y_train_prob=model.predict_proba(x_train)
                    model_train_rocauc_score = roc_auc_score(y_train, y_train_prob,multi_class='ovr',average=average_type) #Calculate Roc Auc Score
            # Test set performance
            else:
                model_test_rocauc_score = np.nan  # Or 0, or skip ROC AUC for this model
            model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
            model_test_f1 = f1_score(y_test, y_test_pred, average=average_type) # Calculate F1-score
            model_test_precision = precision_score(y_test, y_test_pred, average=average_type, zero_division=0) # Calculate Precision
            model_test_recall = recall_score(y_test, y_test_pred, average=average_type,zero_division=0) # Calculate Recall
            if hasattr(model,"predict_proba"):
                if average_type == "binary":
                    y_test_prob = model.predict_proba(x_test)[:,1]
                    model_test_rocauc_score = roc_auc_score(y_test, y_test_prob)
                else:
                    y_test_prob=model.predict_proba(x_test)
                    model_test_rocauc_score = roc_auc_score(y_test, y_test_prob,multi_class='ovr',average=average_type) #Calculate Roc Auc Score
            else:
                model_test_rocauc_score = np.nan  # Or 0, or skip ROC AUC for this model

            model_dict.append({
                "model":list(models.keys())[i],
                "train_accuracy": model_train_accuracy,
                "train_f1": model_train_f1,
                "train_precision": model_train_precision,
                "train_recall": model_train_recall,
                "train_rocauc_score": model_train_rocauc_score,
                "test_accuracy": model_test_accuracy,
                "test_f1": model_test_f1,
                "test_precision": model_test_precision,
                "test_recall": model_test_recall,
                "test_rocauc_score": model_test_rocauc_score,
                "tuned":False
            })
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

    if classification or nlp:
        comparison_df=pd.DataFrame(model_dict)
        comparison_df["total_accuracy"]=comparison_df["train_accuracy"]+comparison_df["test_accuracy"]
        comparison_df["total_f1"]=comparison_df["train_f1"]+comparison_df["test_f1"]
        comparison_df['Norm F1'] = comparison_df["total_f1"] / comparison_df['total_f1'].max()
        comparison_df['Norm Accuracy'] = comparison_df["total_accuracy"] / comparison_df['total_accuracy'].max()
        comparison_df['Combined Score'] = f1_prob * comparison_df['Norm F1'] + (1-f1_prob)* comparison_df['Norm Accuracy']
        combined_score_ranking = comparison_df.sort_values('Combined Score',ascending=False)
        combined_score_ranking=combined_score_ranking[combined_score_ranking["train_f1"]-combined_score_ranking["test_f1"]<overfit_threshold]
        best_models = combined_score_ranking.nlargest(3, 'Combined Score')
    top_model=[i  for i in best_models["model"] ]
    if(len(top_model)==0):
        print("No suitable models found after filtering. Please check your data, model configuration, or relax the overfitting threshold.")
        return
    print("Top Model:",top_model)
    randomcv_model = []
    if regressor:
        from mlforge.params import reg_params
        for i in reg_params:
            if i[0] in top_model:
                randomcv_model.append((i[0], models[i[0]], i[1]))
    if classification :
        from mlforge.params import class_params
        for i in class_params:
            if i[0] in top_model:
                randomcv_model.append((i[0], models[i[0]], i[1]))
        
        # print("Enhanced Classification Model Parameters:", randomcv_model)
    if nlp:
        from mlforge.params import nlp_params
        for i in nlp_params:
             if i[0] in top_model:
                randomcv_model.append((i[0], models[i[0]], i[1]))
    from sklearn.model_selection import RandomizedSearchCV

    model_param = {}
    if classification or nlp:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    if fast:
        n_iter=(int)(n_iter/2)
    for name, model, params in tqdm(randomcv_model, desc="Training best models with enhanced parameters ", unit="model",leave=True,
    dynamic_ncols=True):
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
    if classification or nlp:
        model_cls={
                "LogisticRegression":LogisticRegression,
                "DecisionTreeClassifier":DecisionTreeClassifier,
                "RandomForestClassifier":RandomForestClassifier,
                "GradientBoostingClassifier":GradientBoostingClassifier,
                "AdaBoostClassifier":AdaBoostClassifier,
                "XGBClassifier":XGBClassifier,
                "KNeighborsClassifier":KNeighborsClassifier,
                "SGDClassifier":SGDClassifier
            }
        if nlp:
            model_cls={
                "GaussianNB":GaussianNB,
                "LogisticRegression":LogisticRegression,
                "LinearSVC":LinearSVC,
                "RandomForestClassifier":RandomForestClassifier,
                "XGBClassifier":XGBClassifier,
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
    # print("Training best models with enhanced parameters...")
    if regressor:
        for i in range(len(list(models))):
            model = list(models.values())[i]
            # print("-> "+list(models.keys())[i],flush=True)
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

            model_dict={
                "model":list(models.keys())[i],
                "train_rmse": model_train_rmse,
                        "train_r2": model_train_r2,
                        "test_rmse": model_test_rmse,
                        "test_r2": model_test_r2,
                        "train_mae": model_train_mae,
                        "test_mae": model_test_mae,
                        "train_mse": model_train_mse,
                        "test_mse": model_test_mse,
                        "tuned":True}
            best_models_copy = pd.concat(
        [best_models_copy, pd.DataFrame([model_dict])],
        ignore_index=True
    )

    if classification or nlp:

        for i in range(len(list(models))):
            model = list(models.values())[i]
            # print("-> "+list(models.keys())[i],flush=True)
            model.fit(x_train, y_train) # Train model

            # Make predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Training set performance
            model_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
            model_train_f1 = f1_score(y_train, y_train_pred, average=average_type,zero_division=0) # Calculate F1-score
            model_train_precision = precision_score(y_train, y_train_pred,average=average_type,zero_division=0) # Calculate Precision
            model_train_recall = recall_score(y_train, y_train_pred,average=average_type,zero_division=0) # Calculate Recall
            if hasattr(model,"predict_proba"):
                if average_type == "binary":
                    y_train_prob = model.predict_proba(x_train)[:,1]
                    model_train_rocauc_score = roc_auc_score(y_train, y_train_prob)
                else:
                    y_train_prob=model.predict_proba(x_train)
                    model_train_rocauc_score = roc_auc_score(y_train, y_train_prob, multi_class='ovr', average=average_type)
            # Test set performance
            else:
                model_train_rocauc_score = np.nan  # Or 0, or skip ROC AUC for this model
            model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
            model_test_f1 = f1_score(y_test, y_test_pred, average=average_type,zero_division=0) # Calculate F1-score
            model_test_precision = precision_score(y_test, y_test_pred,average=average_type,zero_division=0) # Calculate Precision
            model_test_recall = recall_score(y_test, y_test_pred,average=average_type,zero_division=0) # Calculate Recall
            if hasattr(model,"predict_proba"):
                if average_type == "binary":
                    y_test_prob = model.predict_proba(x_test)[:,1]
                    model_test_rocauc_score = roc_auc_score(y_test, y_test_prob)
                else:
                    y_test_prob=model.predict_proba(x_test)
                    model_test_rocauc_score = roc_auc_score(y_test, y_test_prob, multi_class='ovr', average=average_type)
            else:
                model_test_rocauc_score = np.nan  # Or 0, or skip ROC AUC for this model
            model_dict={
            "model":list(models.keys())[i],
            "train_accuracy": model_train_accuracy,
            "train_f1": model_train_f1,
            "train_precision": model_train_precision,
            "train_recall": model_train_recall,
            "train_rocauc_score": model_train_rocauc_score,
            "test_accuracy": model_test_accuracy,
            "test_f1": model_test_f1,
            "test_precision": model_test_precision,
            "test_recall": model_test_recall,
            "test_rocauc_score": model_test_rocauc_score,
            "tuned":True
        }
            best_models_copy = pd.concat(
        [best_models_copy, pd.DataFrame([model_dict])],
        ignore_index=True
    )

    if regressor:
        best_models_copy["total_rmse"]=best_models_copy["train_rmse"]+best_models_copy["test_rmse"]
        best_models_copy["total_r2"]=best_models_copy["train_r2"]+best_models_copy["test_r2"]
        best_models_copy['Norm RMSE'] = best_models_copy["total_rmse"] / best_models_copy['total_rmse'].max()
        best_models_copy['Norm R2'] = 1 - (best_models_copy["total_r2"] / best_models_copy['total_r2'].max())
        best_models_copy['Combined Score'] = rmse_prob * best_models_copy['Norm RMSE'] + (1-rmse_prob)* best_models_copy['Norm R2']
        combined_score_ranking = best_models_copy.sort_values('Combined Score').reset_index(drop=True)
    if classification or nlp:
        best_models_copy["total_accuracy"]=best_models_copy["train_accuracy"]+best_models_copy["test_accuracy"]
        best_models_copy["total_f1"]=best_models_copy["train_f1"]+best_models_copy["test_f1"]
        best_models_copy['Norm F1'] = best_models_copy["total_f1"] / best_models_copy['total_f1'].max()
        best_models_copy['Norm Accuracy'] = best_models_copy["total_accuracy"] / best_models_copy['total_accuracy'].max()
        best_models_copy['Combined Score'] = f1_prob * best_models_copy['Norm F1'] + (1-f1_prob)* best_models_copy['Norm Accuracy']
        combined_score_ranking = best_models_copy.sort_values('Combined Score',ascending=False).reset_index(drop=True)


    if classification or nlp:
        combined_score_ranking=combined_score_ranking[combined_score_ranking["train_f1"]-combined_score_ranking["test_f1"]<overfit_threshold]
    best_model_name = combined_score_ranking.iloc[0]["model"]
    best_param_dict = None
    for name, params in best_params:
        if name == best_model_name:
            best_param_dict = params
            break
    model = model_cls[best_model_name](**best_param_dict)
    model.fit(x_train, y_train)
    print("Saving the model , preprocessor...")
    model_path = os.path.join(artifacts_path, "model.pkl")
    preprocessor_path = os.path.join(artifacts_path, "preprocessor.pkl")
    to_save={
        "model":model,
        "dependent_feature":dependent_feature,
        "drop_col":list(dropcorr)
    }
    with open(model_path, "wb") as f:
        pickle.dump(to_save, f)

    if not nlp:
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)
    encoder_path = None
    from mlforge.plots import plot_regression_metrics,plot_classification_metrics,feature_importance,create_cloud
    if classification or nlp:
        if encode:
            encoder_path = os.path.join(artifacts_path, "encoder.pkl")
            with open(encoder_path, "wb") as f:
                pickle.dump(le, f)
        response = {
        "Message": "Training completed successfully",
        "Problem type":"Classification",
            "Model": combined_score_ranking.iloc[0]["model"],
            "Output feature": dependent_feature,
            "Categorical features": cat_features,
            "Numerical features": num_features,
            "Train accuracy": round(float(combined_score_ranking.iloc[0]["train_accuracy"]),4),
            "Train F1": round(float(combined_score_ranking.iloc[0]["train_f1"]),4),
            "Train precision": round(float(combined_score_ranking.iloc[0]["train_precision"]),4),
            "Train recall": round(float(combined_score_ranking.iloc[0]["train_recall"]),4),
            "Train rocauc": round(float(combined_score_ranking.iloc[0]["train_rocauc_score"]),4),
            "Test accuracy": round(float(combined_score_ranking.iloc[0]["test_accuracy"]),4),
            "Test F1": round(float(combined_score_ranking.iloc[0]["test_f1"]),4),
            "Test precision": round(float(combined_score_ranking.iloc[0]["test_precision"]),4),
            "Test recall": round(float(combined_score_ranking.iloc[0]["test_recall"]),4),
            "Test rocauc": round(float(combined_score_ranking.iloc[0]["test_rocauc_score"]),4),
            "Hyper tuned": bool(combined_score_ranking.iloc[0]["tuned"]),
            "Dropped Columns":list(dropcorr)
    }
        if nlp: 
            response["Problem type"]="NLP"
            plots+=create_cloud(df)
        if(response["Hyper tuned"]):
            response["Best Params"] = best_param_dict
        plots+=plot_classification_metrics(model,x_train, y_train, x_test, y_test)
    else:
        response={
            "Message": "Training completed successfully",
            "Problem type": "Regression",
            "Model": combined_score_ranking.iloc[0]["model"],
            "Output feature": dependent_feature,
            "Categorical features": cat_features,
            "Numerical features": num_features,
            "Train R2": round(float(combined_score_ranking.iloc[0]["train_r2"]),4),
            "Train RMSE": round(float(combined_score_ranking.iloc[0]["train_rmse"]),4),
            "Train MAE": round(float(combined_score_ranking.iloc[0]["train_mae"]),4),
            "Train MSE": round(float(combined_score_ranking.iloc[0]["train_mse"]),4),
            "Test R2": round(float(combined_score_ranking.iloc[0]["test_r2"]),4),
            "Test RMSE": round(float(combined_score_ranking.iloc[0]["test_rmse"]),4),
            "Test MAE": round(float(combined_score_ranking.iloc[0]["test_mae"]),4),
            "Test MSE": round(float(combined_score_ranking.iloc[0]["test_mse"]),4),
            "Hyper tuned": bool(combined_score_ranking.iloc[0]["tuned"]),
            "Dropped Columns":list(dropcorr)
        }
        if(response["Hyper tuned"]):
            response["Best Params"] = best_param_dict
        plots+=plot_regression_metrics(model, x_train, y_train, x_test, y_test,feature_names)
    print("\n")
    print("="*55)
    for i in response:
        print(f"{i}: {response[i]}")
    print("="*55)
    print("\n")
    arguments = {
        "data_path": data_path,
        "dependent_feature": dependent_feature,
        "rmse_prob": rmse_prob,
        "f1_prob": f1_prob,
        "n_jobs": n_jobs,
        "n_iter": n_iter,
        "n_splits": n_splits,
        "fast": fast,
        "artifacts_dir": artifacts_dir,
        "artifacts_name": artifacts_name,
        "corr_threshold": corr_threshold,
        "skew_threshold": skew_threshold,
        "z_threshold": z_threshold,
        "overfit_threshold": overfit_threshold,
    }
    if not nlp:
        plots+=feature_importance(model, feature_names)
    generate_dashboard(plots,dashboard_path=os.path.join(artifacts_path, "Dashboard.html"),metrics=response,arguments=arguments,title=dashboard_title,model_comparison_df=best_models_copy)
    print("artifacts_path:", artifacts_path)
    print("model_path:", model_path)
    if not nlp : print("preprocessor_path:", preprocessor_path)
    if encoder_path:
        print("encoder_path:", encoder_path)
    shutil.rmtree(temp_path)
    # return {"status": "success", "model": "trained_model.pkl"}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to the dataset CSV file")
    parser.add_argument("--dependent_feature", required=True, help="Name of the dependent feature (target variable)")
    parser.add_argument("--rmse_prob", type=float, required=True, help="RMSE probability threshold")
    parser.add_argument("--f1_prob", type=float, required=True, help="F1 score probability threshold")
    parser.add_argument("--n_jobs", default=-1, type=int, help="Number of jobs to run in parallel")
    parser.add_argument("--n_iter", default=100, type=int, help="Number of iterations for hyperparameter tuning")
    parser.add_argument("--n_splits", default=5, type=int, help="Number of splits for cross-validation")
    parser.add_argument("--test_size", default=0.2, type=float, help="Proportion of the dataset to include in the test split")
    parser.add_argument("--fast", action="store_true", default=False,
                       help="Enable fast mode for hyperparameter tuning (skip exhaustive tuning).")
    parser.add_argument("--artifacts_dir", default=None, help="Path to save the artifacts")
    parser.add_argument("--artifacts_name", default="artifacts", help="Name of the artifacts directory")
    parser.add_argument("--corr_threshold", type=float, default=0.85, help="Maximum threshold for feature selection")
    parser.add_argument("--skew_threshold", type=float, default=1.0, help="Skewness threshold for feature selection")
    parser.add_argument("--z_threshold", type=float, default=3.0, help="Z-score threshold for outlier removal")
    parser.add_argument("--overfit_threshold", type=float, default=0.15, 
                        help="If the difference between training and test F1 score exceeds this value, "
                             "the model is flagged as overfitting")
    parser.add_argument("--nlp", action="store_true", default=False,help="Enable NLP/text-mode processing (combine text cols, preprocess, vectorize).")
    parser.add_argument("--dashboard_title", type=str, default="mlforgex Dashboard", help="Title of the dashboard")
    args = parser.parse_args()

    train_model(
        data_path=args.data_path,
        dependent_feature=args.dependent_feature,
        rmse_prob=args.rmse_prob,
        f1_prob=args.f1_prob,
        n_jobs=args.n_jobs,
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        test_size=args.test_size,
        fast=args.fast,
        artifacts_dir=args.artifacts_dir,   
        artifacts_name=args.artifacts_name,
        corr_threshold=args.corr_threshold,
        skew_threshold=args.skew_threshold,
        z_threshold=args.z_threshold,
        overfit_threshold=args.overfit_threshold ,
        nlp=args.nlp,
        dashboard_title=args.dashboard_title
    )

