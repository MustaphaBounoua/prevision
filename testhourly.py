# In[
import package as pck
data = pck.getHourlyData()


def accuracy(result):
    pck.getaccuracy(result,1)
    pck.getaccuracy(result,2)
    pck.getaccuracy(result,3)
    pck.getaccuracy(result,4)
    pck.getaccuracy(result,7)

# In[

result365 = pck.cross_validation_arimax(data.nb,'1/7/2018',nbjour_max=180,step=8,horizon=3,train_size=[365])
print ('---------------------365')
accuracy(result365)

result180 = pck.cross_validation_arimax(data.nb,'1/7/2018',nbjour_max=180,step=8,horizon=3,train_size=[180])

print ('---------------------180')
accuracy(result180)

result90 = pck.cross_validation_arimax(data.nb,'1/7/2018',nbjour_max=180,step=8,horizon=3,train_size=[90])
print ('---------------------90')
accuracy(result90)




# In[
d=data['1/3/2018':'7/2/2018']

model1 =pck.sarimax(d.nb,3,0,3,0,1,2,24,pck.getexplanatoryvariables(d.nb))

# In[
model1 =pck.sarima(d.nb,1,0,1,0,1,1,24)

model3 =pck.sarima(d.nb,1,0,1,0,1,2,24)

model4 =pck.sarima(d.nb,3,0,3,0,1,1,24)

model5 =pck.sarima(d.nb,3,0,3,2,1,1,24)

model6 =pck.sarima(d.nb,3,0,3,2,1,2,24)


# In[
import matplotlib.pyplot as plt

model1.plot_diagnostics(figsize=(15, 12))
plt.show()
# In[

# In[


# In[
import time
start_time = time.time()
precision = pck.test_arima(data.nb, '1/7/2018', 4, 365, 3, 0, 3, 0, 1, 2, 24)
print("--- %s seconds ---" % (time.time() - start_time))

# In[

def test_arima(model,data,testdate,nbjourpred,nbjourtest,p=3,d=0,q=3,P=0,D=1,Q=2,s=24):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    horizon = nbjourpred
    nbjourtest = nbjourtest
    from datetime import timedelta
    from pandas import datetime
    test_date_time = datetime.strptime(testdate, '%d/%m/%Y')
    end_test = test_date_time + timedelta(days=horizon)
    end_train = test_date_time
    start_train = datetime.strptime('1/3/2018', '%d/%m/%Y')


    train = data['1/3/2018':'7/1/2018']
    print(train)
    train.drop(train.tail(1).index, inplace=True)

    test = data['7/3/2018':'7/9/2018']
    print(test)
    #test.drop(test.tail(1).index, inplace=True)
    arima_model = model
    ##### horizon 7 journée
    d = data['3/1/2018':'7/2/2018']
    prevision = arima_model.predict(start=model1.fittedvalues.shape[0], end=model1.fittedvalues.shape[0] - 1 + 7 * 24, exog=pck.getexplanatoryvariables(test))
    m=data['6/28/2018':'7/9/2018']
    print(prevision)
    plt.plot(m)
    plt.plot(prevision,'r-')

    plt.show()



test_arima(model1,data['3/1/2018':'9/7/2018'].nb,'7/2/2018',90,24*7)



# In[
from statsmodels.stats.diagnostic import acorr_ljungbox
x=acorr_ljungbox(model1.resid,lags=48,boxpierce=False)
# In[
from statsmodels.stats.stattools import jarque_bera
score, pvalue, _, _ = jarque_bera(model1.resid)
if pvalue < 0.10:
    print ('The residuals may not be normally distributed.')
else:
    print ('The residuals seem normally distributed.')