# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import *
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
import warnings
import wandb
import os
from wandb.keras import WandbCallback
wandb.init(project="forecast-with-cnn")

warnings.filterwarnings("ignore")

wandb.config.dropout = 0.2
wandb.config.batch_size = 6
wandb.config.epochs = 90

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
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=wandb.config.batch_size)


# %%
# 1-d conv

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=(n_input, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(wandb.config.dropout))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse') 
# validation_split=0.33
# #metrics=['accuracy', 'loss', 'val_accuracy', 'val_loss'],)

# %%
model.fit(generator,epochs=wandb.config.epochs,callbacks=[WandbCallback()])
model.save(os.path.join(wandb.run.dir, "cnn-model.h5"))

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
#for (line_number, (index, row)) in enumerate(df_test.iterrows()):
    #wandb.log({'epoch': epoch, 'loss': loss})
    #experiment.log_metric('test', row['AirPassengers'], row['Prediction'], x=line_number)

#experiment.add_output_content('df_test.csv', df_test.to_csv())
df_test.to_csv(os.path.join(wandb.run.dir, "df_test.csv"))
#wandb.save('df_test.csv')
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
wandb.log({'rmse': pred_actual_rmse})
#experiment.log_metric('rmse', pred_actual_rmse)

