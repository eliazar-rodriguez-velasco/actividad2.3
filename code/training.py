#LIBRERIAS
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from joblib import dump, load

dataframe = pd.read_csv("datos00.csv")
df_x = dataframe[['x']]
df_y = dataframe[['y']]

plt.plot(df_x,df_y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("REGRESION L")
plt.savefig("r_lineal.png")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(df_x,df_y, test_size=0.2, random_state=0)

print ('Training: %d rows\nTest: %d rows' % (x_train.shape[0], x_test.shape[0]))

model = LinearRegression()
model.fit(x_train, y_train)

y_hat = model.predict(x_test)
acc = r2_score(y_test, y_hat)
print("Accuracy: %.2f" % acc)
 
dump(model, "model.joblib")
