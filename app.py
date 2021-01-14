import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
# from test.py import best_fit_distribution,  make_pdf
# !pip install plotly
import plotly.express as px

import base64


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_png_as_page_bg('background.png')





# st.set_wide_mode()
st.sidebar.title('Navigation')

df1 = pd.read_csv('Processed.csv')
st.balloons()
st.info("""## GoodReads Data Visulisation""")
# st.markdown("<font color=‘blue’>THIS TEXT WILL BE RED</font>", unsafe_allow_html=True)
st.info("""#### Data collected from the goodreads.com for the dystopian and post-apocalypstic fiction""")

st.markdown('<font color=‘black’>1. Preprocessed Data</font>', unsafe_allow_html=True)

df1 = pd.read_csv('Processed.csv')
df1['Pages'] = df1['Pages'].replace('not known', 0)
df1['Pages'] = df1['Pages'].astype(int)
# df1

# """ Pre Processing Data"""
Scaler = MinMaxScaler((1,10))
# df1['minmax_norm_rating'] = Scaler.fit_transform(df1[['Average rating']])
# df1['minmax_norm_rating1'] = 1 + (df1['Average rating']-df1['Average rating'].min()/df1['Average rating'].max()-df1['Average rating'].min())*9
# df1['meannorm_rating'] = 1 + (df1['Average rating'] - df1['Average rating'].mean())/(df1['Average rating'].max() - df1['Average rating'].min())*9

## Analysis
bygroup_minmax_df1 = pd.DataFrame(data=df1.groupby('Publish year')['minmax_norm_ratings'].agg('mean'))


st.dataframe(df1.style.set_properties(**{'background-color': 'white',
                           'color': 'black',
                           'border-color': 'black'}))


# """Data Visualisation"""

st.info("""### Exercise 1""")

st.markdown('<font color=‘black’>Create a 2D scatterplot with pages on the x-axis and num_ratings on the y-axis.</font>', unsafe_allow_html=True)


# fig, ax = plt.subplots()
# # fig.figure(figsize=(5,4))
# ax.scatter(df1['Pages'], df1['Number of ratings'])
# plt.title("Number of ratings in function of number of pages") # Number of ratings in function of number of pages
# plt.xlabel("Number of pages")
# plt.ylabel("Number of ratings")
# # fig.update_layout(width=1000, height=1000)
fig = px.scatter(df1, x='Pages', y='Number of ratings', title='Number of ratings in function of number of pages',
                 labels={
                     "Pages": "Number of Pages",
                     "Number of Ratings": "Sepal Width (cm)",
                     },
                 )
# fig.show()
st.plotly_chart(fig)


"""
### Exercise 2
Can you compute numerically the correlation coefficient of these two columns?
"""

corr = df1.corr()
st.write('Numerical Correlation for the all the columns:')
# st.text(corr)
st.write(corr.style.background_gradient(cmap='coolwarm'))

# fig, ax = plt.subplots()
# ax.matshow(corr)
# # st.pyplot(fig)


"""
### Exercise 3
Visualise the avg_rating distribution.
"""
average = df1['Average rating']
fig,ax = plt.subplots(figsize=(10,8))
ax.hist(average, bins=(np.arange(1,5,0.1)))
ax.set_ylabel('Rating counts')
a1 = ax.twinx()
a1.hist(average, bins=(np.arange(1,5,0.1)), density= True)
ax.set_xlim(2.5,5)
ax.set_xlabel('Average rating')
average.plot(kind='kde')

st.pyplot(fig)


"""
### Exercise 4
Visualise the minmax_norm_rating distribution.
"""
minmax = df1['minmax_norm_ratings']
fig,ax = plt.subplots(figsize=(10,8))
ax.hist(minmax, bins=np.arange(1,10,0.25))
ax.set_ylabel('Rating Counts')
a1 = ax.twinx()
a1.hist(minmax, bins=np.arange(1,10,0.25), density=True)
minmax.plot(kind='kde')
ax.set_xlim(0,11)
ax.set_xticks(np.arange(0,11))

st.pyplot(fig)

"""
### Exercise 5
Visualise the mean_norm_rating distribution.
"""

mean = df1['mean_norm_ratings']
fig,ax = plt.subplots(figsize=(10,8))
ax.hist(mean, bins=np.arange(-5,5,0.25))
ax.set_ylabel('Rating Counts')
a1 = ax.twinx()
a1.hist(mean, bins=np.arange(-5,5,0.25), density=True)
mean.plot(kind='kde')
ax.set_xlim(-5,5)
ax.set_xticks(np.arange(-5,6))

st.pyplot(fig)

"""
### Exercise 6
Create one graph that represents in the same figure both minmax_norm_rating and mean_norm_rating distributions.
"""

fig,ax = plt.subplots(figsize=(10,8))
mean = df1['mean_norm_ratings'] + 5.82
minmax = df1['minmax_norm_ratings']
ax.hist(mean, color= 'b', bins=np.arange(1,10,0.25), alpha=0.35, label = 'Mean Norm rating counts shifted')
ax.hist(minmax, color = 'r', bins=np.arange(1,10,0.25), alpha=0.35, label = 'MinMax Norm rating counts')
a1 = ax.twinx()
a1.hist(mean, color= 'b', bins=np.arange(1,10,0.25), alpha=0.35, label = 'Mean Norm rating counts shifted', density=True)
mean.plot(kind='kde', alpha=0.7, color='b')
a1.hist(minmax, color = 'r', bins=np.arange(1,10,0.25), alpha=0.35, label = 'MinMax Norm rating counts', density=True)
minmax.plot(kind='kde', alpha=0.7, color='r')
ax.set_xlim(0,11)
plt.legend()

st.pyplot(fig)
# # def Get_Info(authorname df):
#     author_data = df[df['Author']==authorname]
#     author_info = author_data[author_data.minmax_norm_rating == author_data.minmax_norm_rating.max()]
#     return author_info

"""
### Exercise 8
Visualize the awards distribution in a boxplot and aggregtated bars. Decide which of these representations gives us more information and in which cases they should be used. 
"""

fig,ax = plt.subplots(figsize=(10,8))
plt.figure(figsize=(10,7))
sns.boxplot(y=df1['Awards'])
plt.title('')
st.pyplot(fig)