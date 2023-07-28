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
        self.log('Reading configuration file...')
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
        self.store('exact', exact)

        self.log('Reading training data...')
        df = pd.read_csv(f'{INPUT_DIR}/{input_file}', sep = input_sep)
        self.store('dataframe', df)

        self.log('Preparing initial model...')
        lr = LinearRegression().fit(np.zeros(np.shape(df.drop(columns=target_column))), 
                                    np.zeros(np.shape(df[target_column])))
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
        model.fit(df.drop(columns=target_column), df[target_column])
        self.store('model', model)

        self.log('Scoring model...')
        score = model.score(df.drop(columns=target_column), df[target_column])
        
        self.log(f'Score is: {score}') 
        self.log(f'Coefficients: {model.coef_}') 
        self.log(f'Intercept: {model.intercept_}') 

        exact = self.load('exact')
        if exact:
            local_weight = df[target_column].shape[0] # weight of each client
        else:
            delta = random.uniform(-0.1, 0.1)
            local_weight = df[target_column].shape[0] + df[target_column].shape[0]*delta

        weighted_coef = [coef * (local_weight) for coef in model.coef_]
        weighted_intercept = model.intercept_ * (local_weight)
        #self.log(f'Coefficients: {weighted_coef}') 
        #self.log(f'Intercept: {weighted_intercept}') 

        self.send_data_to_coordinator(np.hstack([weighted_coef, weighted_intercept, local_weight]))

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
        self.log('Aggregating model...')
        total_weight = (param[-1])
        agg_coef = (param[0:-3]) / total_weight
        self.store('agg_coef', agg_coef)
        agg_intercept = (param[-2]) / total_weight
        self.store('agg_intercept', agg_intercept) 

        self.log(f'aggregated coefficients: {agg_coef}, \
                 \naggregated intercept: {agg_intercept}')

        done = self.load('iteration') >= self.load('max_iterations')

        self.log('Broadcasting gloabel model...')
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
        input_sep = self.load('input_sep')
        target_column = self.load('target_column')
        output_file = self.load('output_file')

        test_df = pd.read_csv(f'{INPUT_DIR}/{input_test}', sep=input_sep)
        self.store('dataframe', test_df)

        if target_column in test_df.columns:
            #column_one = np.ones((test_df.shape[0], 1)).astype(np.uint8)
            X_test = test_df.drop(columns=[target_column])
            #X_test = np.concatenate((column_one, X_test), axis=1)
        else:
            self.log(f'Target column "{target_column}" not found in the test data.')

        #print(X_test)

        pd.DataFrame(data={'coef': [model.coef_], 'intercept': [model.intercept_],
                        'pred': [model.predict(X_test)]}) \
            .to_csv(f'{OUTPUT_DIR}/{output_file}', index=False)

        return TERMINAL_STATE

