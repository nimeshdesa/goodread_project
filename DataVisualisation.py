import base64

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from fitter import Fitter

class datavisualisation():
    def __init__(self):
        self.df1 = pd.read_csv('Processed.csv')
        self.groupby = self.groupby()
        self.pages_to_numeric()

    def get_info(self,authorname):
        self.author_data = self.df1[self.df1['Author'].isin([authorname])]
        self.author_info = self.author_data[self.author_data.minmax_norm_ratings == self.author_data.minmax_norm_ratings.max()]
        return self.author_info

    def get_base64_of_bin_file(self,bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def set_png_as_page_bg(self,png_file):
        bin_str = self.get_base64_of_bin_file(png_file)
        page_bg_img = '''
        <style>
        body {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        opacity: 1;
        }
        </style>
        ''' % bin_str
        return bin_str





    def groupby(self):
        self.groupby = pd.DataFrame(data=self.df1.groupby('Publish year')['minmax_norm_ratings'].agg('mean'))
        return self.groupby


    def pages_to_numeric(self):
        self.df1['Pages'] = self.df1['Pages'].apply(lambda x: x.replace("not known", "0"))
        self.df1['Pages'] = pd.to_numeric(self.df1['Pages'])

    # ### Exercise 1
    # Create a 2D scatterplot with pages on the x-axis and num_ratings on the y-axis.
    def pages_NumRatings(self):
        fig = px.scatter(x=self.df1['Pages'], y=self.df1['Number of ratings'], title='Number of ratings in function of number of pages',
                 labels={
                     "Pages": "Number of Pages",
                     "Number of Ratings": "Sepal Width (cm)",
                     })
        return fig

    # ### Exercise 2
    # Can you compute numerically the correlation coefficient of these two columns?
    def correlation_coeffi(self):
        self.corr = self.df1.corr()
        return self.corr.style.background_gradient(cmap='coolwarm')

    # ### Exercise 3
    # Visualise the avg_rating distribution.
    def avg_rating_dist(self):

        average = self.df1['Average rating']
        fig,ax = plt.subplots(figsize=(10,8))
        ax.hist(average, bins=(np.arange(1,5,0.1)))
        ax.set_ylabel('Rating counts')
        a1 = ax.twinx()
        a1.hist(average, bins=(np.arange(1,5,0.1)), density= True)
        ax.set_xlim(2.5,5)
        ax.set_xlabel('Average rating')
        average.plot(kind='kde')
        return fig

    # ### Exercise 4
    # Visualise the minmax_norm_rating distribution.
    def minmax_norm_ratings_dist(self):

        minmax = self.df1['minmax_norm_ratings']
        fig,ax = plt.subplots(figsize=(10,8))
        ax.hist(minmax, bins=np.arange(1,10,0.25))
        ax.set_ylabel('Rating Counts')
        a1 = ax.twinx()
        a1.hist(minmax, bins=np.arange(1,10,0.25), density=True)
        minmax.plot(kind='kde')
        ax.set_xlim(0,11)
        ax.set_xticks(np.arange(0,11))
        plt.show()

    # ### Exercise 5
    # Visualise the mean_norm_rating distribution.
    def mean_norm_ratings(self):
        mean = self.df1['mean_norm_ratings']
        fig,ax = plt.subplots(figsize=(10,8))
        ax.hist(mean, bins=np.arange(-5,5,0.25))
        ax.set_ylabel('Rating Counts')
        a1 = ax.twinx()
        a1.hist(mean, bins=np.arange(-5,5,0.25), density=True)
        mean.plot(kind='kde')
        ax.set_xlim(-5,5)
        ax.set_xticks(np.arange(-5,6))
        plt.show()
        return fig


    # ### Exercise 6
    # Create one graph that represents in the same figure both minmax_norm_rating and mean_norm_rating distributions.
    def mean_minmax_norm_ratings_dist(self):
        fig,ax = plt.subplots(figsize=(10,8))
        mean = self.df1['mean_norm_ratings'] + 5.82
        minmax = self.f1['minmax_norm_ratings']
        ax.hist(mean, color= 'b', bins=np.arange(1,10,0.25), alpha=0.35, label = 'Mean Norm rating counts shifted')
        ax.hist(minmax, color = 'r', bins=np.arange(1,10,0.25), alpha=0.35, label = 'MinMax Norm rating counts')
        a1 = ax.twinx()
        a1.hist(mean, color= 'b', bins=np.arange(1,10,0.25), alpha=0.35, label = 'Mean Norm rating counts shifted', density=True)
        mean.plot(kind='kde', alpha=0.7, color='b')
        a1.hist(minmax, color = 'r', bins=np.arange(1,10,0.25), alpha=0.35, label = 'MinMax Norm rating counts', density=True)
        minmax.plot(kind='kde', alpha=0.7, color='r')
        ax.set_xlim(0,11)
        plt.legend()
        return fig

    def mean_minmax_norm_ratings_dist1(self):
        plt.figure(figsize=(10,7))
        sns.distplot(self.df1['mean_norm_rating'], kde = True, hist = True, rug= False, bins= 40)
        sns.distplot(self.df1['minmax_norm_ratings'], kde = True, hist = True, rug= False, bins= 40)
        plt.xlabel('Mean_MinMax_Norm_Rating', fontsize=10)
        plt.ylabel('Density',fontsize=10)
        return plt


    # ### Exercise 7
    # What is the best fit in terms of a distribution (normal, chi-squared...) to represent each of those graphs?
    # * You can use Scipy-Stats Library to figure out the best fitting distribution.
    def best_fit_avg_rating(self):

        data1 = self.df1['Average rating']
        f = Fitter(data1)
        f.fit()
        f.summary()
        return f

    def best_fit_minmax_nor_raing(self):
        data2 = self.df1['minmax_norm_rating']
        g = Fitter(data2)
        g.fit()
        g.summary()
        return(g)


    # ### Exercise 8
    # Visualize the awards distribution in a boxplot and aggregtated bars.
    # Decide which of these representations gives us more information and in which cases they should be used.
    def box_plot(self):
        fig,ax = plt.subplots(figsize=(10,8))
        plt.figure(figsize=(10,7))
        sns.boxplot(y=self.df1['Awards'])
        plt.title('Awards distribution')
        return plt

    def box_and_agg_bar_plot(self):
        jointplot = sns.JointGrid(data=self.df1, x="Awards")
        jointplot.fig.set_figwidth(9)
        jointplot.fig.set_figheight(9)
        jointplot.plot(sns.histplot, sns.boxplot)
        return jointplot



    # ### Exercise 9
    # Yesterday we asked you this:
    #     * "Group the books by original_publish_year and get the mean of the minmax_norm_ratings of the groups."
    #     * Now, make a simple plot to visualise the ratings w.r.t. the years!
    def rating_wrt_years(self):
        self.groupby.drop(self.groupby.index[-3:], inplace=True)
        fig, ax = plt.subplots(1, sharex=True,figsize=(20,8))
        ax.plot(self.groupby.index, self.groupby['minmax_norm_ratings'], label = 'MinMax_Norm_Ratings', color='b', marker='o', linewidth=3)
        ax.set_title('MinMax_Norm_Ratings_Bygroup')
        plt.xticks(self.groupby.index,  rotation='vertical')
        plt.xlabel('Year', fontsize=20)
        plt.ylabel('MinMax_Norm_Rating_Mean', fontsize=20)
        return fig


    # ### Exercise 10
    # Make a scatterplot to represent  minmax_norm_ratings in function of the number of awards won by the book.
    # - Is there another representation that displays this in a more clear manner?
    # - Optional: Can you plot a best fit linear regression line to represent the relationship?
    def regplot(self):
        plt.figure(figsize=(10,8))
        return sns.regplot(y="minmax_norm_rating", x="Awards", data=df1);






