from FeatureCloud.app.engine.app import AppState, app_state, Role
import bios
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import random


INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'

INITIAL_STATE = 'initial'
COMPUTE_STATE = 'compute'
AGGREGATE_STATE = 'aggregate'
WRITE_STATE = 'write'
TERMINAL_STATE = 'terminal'


@app_state(INITIAL_STATE)
class InitialState(AppState):

    def register(self):
        self.register_transition(COMPUTE_STATE) 

    def run(self):
        print('-------------------------Reading configuration file--------------------')
        config = bios.read(f'{INPUT_DIR}/config.yml')
        max_iterations = config['max_iter']
        input_file = config['train']
        input_test = config['test']
        output_file = config['output']
        input_sep = config['sep']
        target_column = config['target_value']
        exact = config['exact']
        total_weight = 0

        self.store('max_iterations', max_iterations)
        self.log(f'MAX_ITERATIONS: {max_iterations}')
        self.store('target_column', target_column)
        self.store('input_file', input_file)
        self.store('input_test', input_test)
        self.store('input_sep', input_sep)
        self.store('output_file', output_file)
        self.store('total_weight', total_weight)
        self.store('exact',exact)

        self.log('Reading training data...')
        df = pd.read_csv(f'{INPUT_DIR}/{input_file}', sep = input_sep)
        self.store('dataframe', df)

        print('----------------------Preparing initial model--------------------------')
        if target_column is None:
            lr = LinearRegression().fit(np.zeros(np.shape(df.iloc[:,:-1])), 
                                    np.zeros(np.shape(df.iloc[:,-1])))
        else:
            lr = LinearRegression().fit(np.zeros
                                        (np.shape(df.drop(columns=target_column)))
                                        ,np.zeros(np.shape(df[target_column])))
        self.store('model', lr)
        self.store('iteration', 0)

        if self.is_coordinator:
            self.broadcast_data([lr.coef_, lr.intercept_, False])       
            
        return COMPUTE_STATE
    

@app_state(COMPUTE_STATE)
class ComputeState(AppState):

    def register(self):

        self.register_transition(COMPUTE_STATE, role=Role.PARTICIPANT)  
        self.register_transition(AGGREGATE_STATE, role=Role.COORDINATOR)  
        self.register_transition(WRITE_STATE)  

    def run(self):
        iteration = self.load('iteration')
        iteration +=1
        self.store('iteration', iteration)

        self.log(f'ITERATION {iteration}')

        # Recieve global model from coordinator
        coef, intercept, done = self.await_data()

        model = self.load('model')
        model.coef_ = coef
        model.intercept_ = intercept

        if done: 
            return WRITE_STATE
        
        self.log('Fitting model...')
        
        df = self.load('dataframe')

        target_column = self.load('target_column')
        scores = []
        if target_column is None:
            model.fit(df.iloc[:,:-1], df.iloc[:,-1])
            score = model.score(df.iloc[:, :-1], df.iloc[:, -1])
        else:
            model.fit(df.drop(columns=target_column), df[target_column])
            score = model.score(df.drop(columns=target_column), df[target_column])

        scores.append(score)
        self.store('model', model)
        print('----------------------Done woth Local computation----------------------')

        exact = self.load('exact')
        if exact:
            if target_column is None:
                local_weight = df.iloc[:,-1].shape[0] # weight of each client
            else:
                local_weight = df[target_column].shape[0] # weight of each client
        else:
            delta = random.uniform(-0.1, 0.1)
            if target_column is None:
                local_weight = df.iloc[:,-1].shape[0] + df.iloc[:,-1].shape[0]*delta
            else:
                local_weight = df[target_column].shape[0] + df[target_column].shape[0]*delta

        weighted_coef = [coef * (local_weight) for coef in model.coef_]
        weighted_intercept = model.intercept_ * (local_weight)

        self.send_data_to_coordinator(np.hstack([weighted_coef, weighted_intercept, local_weight]))
        print('--------------------local betas sent for aggregation------------------')
        
        if self.is_coordinator:
            return AGGREGATE_STATE
        else:
            return COMPUTE_STATE

    
@app_state(AGGREGATE_STATE)
class AggregateState(AppState):

    def register(self):
        self.register_transition(COMPUTE_STATE, role=Role.COORDINATOR) 

    def run(self):
        self.log('Waiting for local models...')
        param = self.aggregate_data()
        # print("param", param)
        print('-----------------------Aggregating model...---------------------')
        total_weight = (param[-1])
        agg_coef = (param[0:-2]) / total_weight
        self.store('agg_coef', agg_coef)
        agg_intercept = (param[-2]) / total_weight
        self.store('agg_intercept', agg_intercept) 

        # self.log(f'aggregated coefficients: {agg_coef}, \
        #          \naggregated intercept: {agg_intercept}')

        done = self.load('iteration') >= self.load('max_iterations')

        print('---------------------Broadcasting gloabel model...--------------------')
        self.broadcast_data([agg_coef, agg_intercept, done])

        return COMPUTE_STATE  
    

@app_state(WRITE_STATE)
class WriteState(AppState):
    def register(self):
        self.register_transition(TERMINAL_STATE)  

    def run(self):
        self.log('Writing data to file...')
        model = self.load('model')
        input_test = self.load('input_test')
        input_train = self.load('input_file')  # train is named as input_file
        input_sep = self.load('input_sep')
        target_column = self.load('target_column')

        test_df = pd.read_csv(f'{INPUT_DIR}/{input_test}', sep=input_sep)
        self.store('dataframe', test_df)

        train_df = pd.read_csv(f'{INPUT_DIR}/{input_train}', sep=input_sep)
        self.store('dataframe', train_df)

        if target_column in test_df.columns:
            X_test = test_df.drop(columns=[target_column])
            Y_test = test_df[target_column] 
        else:
            X_test = test_df.iloc[:,:-1]
            Y_test = test_df.iloc[:,-1]

        if target_column in train_df.columns:
            X_train = train_df.drop(columns=[target_column])
            Y_train = train_df[target_column] 
        else:
            X_train = train_df.iloc[:,:-1]
            Y_train = train_df.iloc[:,-1]

        #print evaluation
        y_pred = model.predict(X_test)
        print("Train Accuracy:", model.score(X_train,Y_train), "Test Accuracy:", model.score(X_test,Y_test))
        print ("test r_sq:", 1 - np.sum( (np.array(Y_test)- np.array(y_pred))**2 )/np.sum ((np.array(Y_test)-np.mean(np.array(Y_test)))**2 ))
        print ("test RMSE:", ( np.sum((np.array(Y_test)- np.array(y_pred))**2) / np.array(Y_test).shape[0])**0.5 )
        print ("test NRMSE:", ( (np.sum((np.array(Y_test)- np.array(y_pred))**2) / np.array(Y_test).shape[0])**0.5)/np.mean(np.array(Y_test)) )
        print ("coef: ", model.coef_, "intercept: ", model.intercept_)
                
        print ("----------------------Finished------------------")
        return TERMINAL_STATE

