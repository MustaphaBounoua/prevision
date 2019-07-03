# In[
import dailypackage as pck
datahourly = pck.getHourlyData()
datadily=pck.getHourlyData(daily=True)

def accuracy(result):
    pck.getaccuracy(result,1)
    pck.getaccuracy(result,2)
    pck.getaccuracy(result,3)
    pck.getaccuracy(result,4)
    pck.getaccuracy(result,7)


result90 = pck.cross_validation_arimax(datahourly,datadily.nb,'1/7/2018',nbjour_max=180,step=8,horizon=3,train_size=[90])
print ('---------------------365')
accuracy(result90)



# In[

result365 = pck.cross_validation_arimax(datahourly,datadily.nb,'1/7/2018',nbjour_max=180,step=8,horizon=3,train_size=[365])
print ('---------------------365')
accuracy(result365)

result180 = pck.cross_validation_arimax(datahourly,datadily.nb,'1/7/2018',nbjour_max=180,step=8,horizon=3,train_size=[180])
print ('---------------------180')
accuracy(result180)




# In[

# In[
result90 = pck.cross_validation_arimax(datahourly,datadily.nb,'1/7/2018',nbjour_max=10,step=8,horizon=3,train_size=[90])
print ('---------------------90')
accuracy(result90)


# In[
('The residuals seem normally distributed.')