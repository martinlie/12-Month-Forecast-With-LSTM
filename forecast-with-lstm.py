# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import warnings
import deepkit

experiment = deepkit.experiment()
experiment.add_file(__file__)

warnings.filterwarnings("ignore")

df = pd.read_csv('AirPassengers.csv')


# %%
df.Month = pd.to_datetime(df.Month)
df = df.set_index("Month")


# %%
train, test = df[:-12], df[-12:]


# %%
scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)


# %%
n_input = 12
n_features = 1
batch_size = experiment.intconfig('batch_size')
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=batch_size)


# %%
# 1-d conv

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features))) #, return_sequences=True))
#model.add(LSTM(200, activation='relu'))
model.add(Dropout(experiment.floatconfig('dropout')))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse') 
# validation_split=0.33
# #metrics=['accuracy', 'loss', 'val_accuracy', 'val_loss'],)

experiment.watch_keras_model(model, model_input=next(iter(generator)), is_batch=True)
deepkit_callback = experiment.create_keras_callback() 

# %%
model.fit(generator,epochs=90,callbacks=[deepkit_callback])


# %%
pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)


# %%
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=df[-n_input:].index, columns=['Prediction'])

df_test = pd.concat([df,df_predict], axis=1)


# %%
for (line_number, (index, row)) in enumerate(df_test.iterrows()):
    experiment.log_metric('test', row['AirPassengers'], row['Prediction'], x=line_number)

experiment.add_output_content('df_test.csv', df_test.to_csv())
#experiment.log_insight(df_test.to_numpy(), name='results/test', meta='Test data', image_convertion=False)

#plt.figure(figsize=(20, 5))
#plt.plot(df_test.index, df_test['AirPassengers'])
#plt.plot(df_test.index, df_test['Prediction'], color='r')
#plt.legend(loc='best', fontsize='xx-large')
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=16)
#plt.show()


# %%
pred_actual_rmse = rmse(df_test.iloc[-n_input:, [0]], df_test.iloc[-n_input:, [1]])
print("rmse: ", pred_actual_rmse)
experiment.log_metric('rmse', pred_actual_rmse)

exit()

# %%
train = df


# %%
scaler.fit(train)
train = scaler.transform(train)


# %%
n_input = 12
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)


# %%
model.fit_generator(generator,epochs=90)


# %%
pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)


# %%
from pandas.tseries.offsets import DateOffset
add_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,13) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)


# %%
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-n_input:].index, columns=['Prediction'])

df_proj = pd.concat([df,df_predict], axis=1)


# %%
plt.figure(figsize=(20, 5))
plt.plot(df_proj.index, df_proj['AirPassengers'])
plt.plot(df_proj.index, df_proj['Prediction'], color='r')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()


