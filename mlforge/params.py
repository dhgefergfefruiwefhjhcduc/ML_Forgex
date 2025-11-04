import numpy as np
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
                'kernel': ['rbf', 'linear'],
                'C': [0.1, 0.5, 1, 5, 10],
                'epsilon': [0.01, 0.1, 0.2, 0.5, 1.0],
                'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10],
                'degree': [2, 3, 4]  # Only for poly kernel
            }]
        ]



class_params = [
            ["LogisticRegression", {
                'penalty': ['l2'],               
                'C': np.logspace(-4, 4, 20),
                'solver': ['lbfgs', 'sag', 'saga','newton-cg','newton-cholesky'],
                'max_iter': [100, 200, 500, 1000],
                'class_weight': [None, 'balanced'],
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
            
            ["SGDClassifier", {  # Fast, works like linear SVM/LogReg
            'loss': ['hinge', 'log_loss'],
            'penalty': ['l2', 'elasticnet'],
            'alpha': [1e-4, 1e-3, 1e-2],
            'max_iter': [1000, 3000],
            'class_weight': [None, 'balanced'],
        }]

            
        
        ]

nlp_params= [
    ["LogisticRegression", {
        'penalty': ['l2'],               
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [500, 1000],
        'class_weight': [None, 'balanced'],
    }],
    
    ["LinearSVC", { 
        'C': [0.01, 0.1, 1, 10],
        'class_weight': [None, 'balanced'],
        'max_iter': [1000, 3000]
    }],
    
    ["RandomForestClassifier", {
        'n_estimators': [10,20,50,100,200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 10],
        'class_weight': ['balanced'],
        'random_state': [42]
    }],
    
    ["XGBClassifier", {
        'n_estimators': [10,20,50,100,200],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'random_state': [42],
        'eval_metric': ['logloss']
    }],
["GaussianNB", {
    'var_smoothing': np.logspace(-12, -6, 7),   
}]
]


