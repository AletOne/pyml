import linear_regression
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Start")
    df = pd.read_csv("./archive/train.csv")

    testDf = pd.read_csv("./archive/test.csv")
    X_test = testDf.x
    y_true = testDf.y

    df.dropna(inplace=True)
    #print(df)
    X = df["x"].to_numpy().T
    Y = df["y"].to_numpy().T
    # plt.scatter(X, Y, s=10)
    # plt.show()

    lr = linear_regression.LinearRegression()
    lr.fit(X, Y)
    y_pred = lr.predict(X_test)

    print(X_test)
    print("MSE is", linear_regression.mean_squared_error(y_true, y_pred))

    plt.scatter(X_test, y_true,  color='black')#散点输出
    plt.plot(X_test, y_pred, color='blue', linewidth=3)#预测输出
    plt.show()

main()