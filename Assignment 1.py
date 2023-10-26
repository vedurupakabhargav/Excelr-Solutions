#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random


# In[ ]:


###7th Question


# In[2]:


dt = pd.read_csv("Q7.csv")
dt


# In[3]:


dt.describe()


# In[4]:


dt.var()


# In[5]:


dt.std()


# In[ ]:





# In[8]:


dt['Weigh'].min()


# In[9]:


dt['Weigh'].max()


# In[10]:


dt['Points'].min()


# In[11]:


dt['Points'].max()


# In[12]:


dt.median()


# In[ ]:


###9th Question


# In[14]:


A9 = pd.read_csv("Q9_a (1).csv")
A9


# In[15]:


A9.skew()


# In[ ]:





# In[16]:


A9.kurtosis()


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


plt.scatter(A9['speed'], A9['dist'])
plt.xlabel('Speed')
plt.ylabel('Distance')
plt.title('Scatter Plot of Speed vs Distance')
plt.show()


# In[19]:


B9 = pd.read_csv("Q9_b.csv")
B9


# In[20]:


B9.skew()


# In[21]:


B9.kurtosis()


# In[ ]:


###11th Question


# In[59]:


import numpy as np
import pandas as pd 
from scipy import stats
from scipy.stats import norm


# In[76]:


# Avg. weight of Adult in Mexico with 94% CI
stats.norm.interval(0.94,200,30/(2000**0.5))


# In[78]:


# Avg. weight of Adult in Mexico with 96% CI
stats.norm.interval(0.96,200,30/(2000**0.5))


# In[77]:


# Avg. weight of Adult in Mexico with 98% CI
stats.norm.interval(0.98,200,30/(2000**0.5))


# In[ ]:


###12th Question


# In[25]:


s =[34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]
s


# In[26]:


mean = np.mean(s)
mean


# In[27]:


median = np.median(s)
median


# In[28]:


sd= np.std(s)
sd


# In[29]:


var = np.var(s)
var


# In[ ]:


##20th QUestion


# In[30]:


df = pd.read_csv("Cars.csv")
df


# In[31]:


df.std()


# In[32]:


df['MPG'].mean()


# In[33]:


cars = norm(34.422,9.131)
cars


# In[34]:


x2 = cars.cdf(38)
x2


# In[35]:


1-cars.cdf(38)


# In[36]:


cars.cdf(40)


# In[37]:


cars.cdf(50)


# In[38]:


cars.cdf(20)


# In[39]:


df


# In[40]:


mpg_data = df['MPG']
mpg_data


# In[ ]:





# In[43]:


###21(a) question


# In[44]:


from scipy.stats import probplot


# In[45]:


probplot(mpg_data, dist="norm", plot=plt)
plt.title("Normal Probability Plot")
plt.show()


# In[46]:


from scipy.stats import shapiro

stat, p = shapiro(mpg_data)

if p > 0.05:
    print("The data follows a normal distribution.")
else:
    print("The data does not follow a normal distribution.")


# In[ ]:


###21(b) Question


# In[47]:


dr = pd.read_csv("wc-at (1).csv")
dr


# In[48]:


waist_data = dr['Waist']
waist_data


# In[49]:


probplot(waist_data, dist="norm", plot=plt)
plt.title("Normal Probability Plot")
plt.show()


# In[50]:


at_data = dr['AT']
at_data


# In[51]:


probplot(at_data, dist="norm", plot=plt)
plt.title("Normal Probability Plot")
plt.show()


# In[ ]:


###22th question


# In[79]:


from scipy import stats
from scipy.stats import norm 


# In[80]:


# Z-score of 90% confidence interval 
stats.norm.ppf(0.95)


# In[81]:



# Z-score of 94% confidence interval
stats.norm.ppf(0.97)


# In[82]:


# Z-score of 60% confidence interval
stats.norm.ppf(0.8)


# In[ ]:


###23Question


# In[52]:


from scipy import stats
from scipy.stats import norm


# In[83]:


# t scores of 95% confidence interval for sample size of 25
stats.t.ppf(0.975,24)  # df = n-1 = 24


# In[84]:



 # t scores of 96% confidence interval for sample size of 25
stats.t.ppf(0.98,24)


# In[85]:



# t scores of 99% confidence interval for sample size of 25
stats.t.ppf(0.995,24)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




