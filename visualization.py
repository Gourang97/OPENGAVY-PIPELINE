import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

class visualization():
    
    def __init__(self):
        print ("In Visulaization")

    def check_data_types(self, data, x, y):
        if data.loc[:, x].dtypes != np.float64 or data.loc[:, x].dtypes != np.int64:
            x_type = 'categorical'
        else:
            x_type = 'continuous'
        
        if data.loc[:, y].dtypes != np.float64 or data.loc[:, y].dtypes != np.int64:
            y_type = 'categorical'
        else:
            y_type = 'continuous'

        return x_type, y_type

    def open_plot(self, data, x, y = [], dict_options = {}):
        
        if not y:
            if data.loc[:, x].dtypes != np.float64 or data.loc[:, x].dtypes != np.int64:
                x_type = 'categorical'
            else:
                x_type = 'continuous'

            if x_type == 'categorical':
                self.open_bar(data, x, dict_options)
            elif x_type == 'continuous':
                self.open_histogram(data, x, dict_options)
                

        else:
            x_type, y_type = self.check_data_types(data, x, y)

            if x_type == 'categorical' and y_type == 'categorical':
                self.open_scatter(data, x, y, dict_options)
            elif x_type == 'continuous' and y_type == 'categorical':
                self.open_boxplot(data, y, x, dict_options)
            elif x_type == 'categorical' and y_type == 'continuous':
                self.open_boxplot(data, x, y, dict_options)
            elif x_type == 'continuous' and y_type == 'continuous': 
                self.open_line(data, x, y, dict_options)

    def return_dict_options(self, dict_options = {}):
        if 'xlabel' in dict_options:
            xlabel = dict_options['xlabel']
        else:
            xlabel = ''

        if 'ylabel' in dict_options:
            ylabel = dict_options['ylabel']
        else:
            ylabel =''

        if 'title' in dict_options:
            title = dict_options['title']
        else:
            title = "Distribution of {} v/s {}".format('x', 'y')

        if 'xscale' in dict_options:
            xscale = dict_options['xscale']
        else:
            xscale = 'linear'

        if 'yscale' in dict_options:
            yscale = dict_options['yscale']
        else:
            yscale = 'linear'

        return xlabel, ylabel, title, xscale, yscale

    def open_scatter(self, data, x, y, dict_options = {}):
        xlabel, ylabel, title, xscale, yscale = self.return_dict_options(dict_options)
        plt.scatter(x, y, data = data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.show()

    def open_histogram(self, data, x, y, dict_options = {}):
        xlabel, ylabel, title, xscale, yscale = self.return_dict_options(dict_options)
        plt.hist(x, data = data)
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.title(title)
        plt.xscale(xscale)
        plt.yscale(yscale)

    def open_boxplot(self, data, x, y, dict_options = {}):
        xlabel, ylabel, title, xscale, yscale = self.return_dict_options(dict_options)
        plt.boxplot(x, y, data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xscale(xscale)
        plt.yscale(yscale)

    def open_line(self, data, x, y, dict_options = {}):
        xlabel, ylabel, title, xscale, yscale = self.return_dict_options(dict_options)
        plt.plot(x, y, data = data)
        plt.xlabel(xlabel)
        plt.xlabel(ylabel)
        plt.title(title)
        plt.xscale(xscale)
        plt.yscale(yscale)

    def open_bar(self, data, x, dict_options = {}):
        xlabel, ylabel, title, xscale, yscale = self.return_dict_options(dict_options)
        plt.bar(x, data = data)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.xscale(xscale)
        plt.yscale(yscale)

    def open_recommend_plot(self, data, dict_options = {}):
        data_copy = data.copy()
        df_corr = self.corelation(data_copy)
        df_corr = df_corr.sort_values(by = 'TARGET', ascending = False)

        top = 0
        for i in df_corr.reset_index().loc[:, 'index']:
            if top < 5:
                top = top + 1
                self.open_plot(data, i, "TARGET", {})

        
    def corelation(self, df):
        X = df.copy()
        corelation = pd.DataFrame(stats.spearmanr(X)[0])
        corelation.columns = X.columns
        corelation.index = X.columns
        return corelation

    
    