#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math


# In[10]:


a,b,x = map(int, input().split())


# In[13]:


theta = 0
if x >= a**2*b/2:
    tan = (2*(a*a*b-x))/(a**3)
    ans = math.degrees(math.atan(tan))
else:
    tan = (a*b*b)/(2*x)
    ans = math.degrees(math.atan(tan))
print(ans)


# In[ ]:




