
import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline 
import numpy as np

# Latitude and longitude coordinates for your location
latitude = 33.465959
longitude = -112.073502

# Set up the initial start date and end date for the first 5-year window
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365*5)).strftime('%Y-%m-%d')

# Construct the URL with the initial start_date and end_date
url = f'https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m&hourly=dewpoint_2m&hourly=relativehumidity_2m'

# Retrieve the data and create a pandas DataFrame
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data).transpose()
df = df[['time','relativehumidity_2m','dewpoint_2m','temperature_2m']]
df = df.transpose()
df = df[['hourly']].transpose()
df = df[['relativehumidity_2m','dewpoint_2m','temperature_2m','time']].explode(['relativehumidity_2m','dewpoint_2m','temperature_2m','time']) #need to do this for the others as well
df['temperature_2m'] = (df['temperature_2m'] * 1.8) + 32
df['dewpoint_2m'] = (df['dewpoint_2m'] * 1.8) + 32
df['time'] = pd.to_datetime(df['time'])


class humdity_calcs:
    
    def __init__(self, date_col='na', relative_humidity_col='na', dew_point_col='na', temp_col='na', location='na'):
        self.date_col = date_col
        self.relative_humidity_col = relative_humidity_col
        self.dew_point_col = dew_point_col
        self.temp_col = temp_col
        self.location = location

    def mon_lbl(self,row):
        if row == 1:
            return '01'
        elif row == 2:
            return '02'
        elif row == 3:
            return '03'
        elif row == 4:
            return '04'
        elif row == 5:
            return '05'
        elif row == 6:
            return '06'
        elif row == 7:
            return '07'
        elif row == 8:
            return '08'
        elif row == 9:
            return '09'
        else:
            return row

    def comf_scrng(self,row):
        return 10 + (10 * row)

    #MONTHLY AVERAGE HUMIDITY - DONE
    def mon_avg_hum(self):
        humid_mo = df.groupby(df[self.date_col].dt.month)[self.relative_humidity_col].mean()
        humid_mo.plot(y=self.relative_humidity_col, x=self.date_col, kind='line', figsize=(9, 8))
        return plt.show()

    #MONTHLY AVERAGE DEW POINT - DONE
    def mon_avg_dp(self):
        dewpoint_mo = df.groupby(df[self.date_col].dt.month)[self.dew_point_col].mean()
        dewpoint_mo.plot(y=self.dew_point_col, x=self.date_col, kind='line', figsize=(9, 8))
        return plt.show()

    #DAILY ANNUAL HUMIDITY (NEED TO HAVE THIS JUST BE ON AN ANNUAL BASIS) - DONE
    def daily_hum(self,level):
        d_humid = df
        
        #date_type can be day, month, etc.
        if level == 'mean':
            d_humid = df.groupby(pd.Grouper(key = self.date_col, freq = 'D'))[self.relative_humidity_col].mean()
            d_humid = pd.DataFrame(d_humid)
            d_humid = d_humid.reset_index()
            d_humid['month'] = d_humid[self.date_col].dt.month
            d_humid['month'] = d_humid['month'].apply(self.mon_lbl)
            d_humid['month'] = d_humid['month'].astype(str)
            d_humid['day'] = d_humid[self.date_col].dt.day
            d_humid['day'] = d_humid['day'].apply(self.mon_lbl)
            d_humid['day'] = d_humid['day'].astype(str)
            d_humid['combo'] = d_humid['month'] + d_humid['day']
            d_humid = d_humid.groupby(['combo'],as_index=False)[self.relative_humidity_col].mean()
            #d_humid = df.groupby(['day'], as_index = False)[self.relative_humidity_col].mean()
            #d_humid.plot(y= self.relative_humidity_col, x='combo', kind='line', figsize=(9, 8))
            return print(d_humid)

        elif level == 'max':
            d_humid = df.groupby(['day'], as_index = False)[self.relative_humidity_col].max()
            #d_humid.plot(y=self.relative_humidity_col, x=self.date_col, kind='line', figsize=(9, 8))
            return print(d_humid)

        elif level == 'min':
            d_humid = df.groupby(['day'], as_index = False)[self.relative_humidity_col].min()
            #d_humid.plot(y=self.relative_humidity_col, x=self.date_col,kind='line', figsize=(9, 8))
            return print(d_humid)

        else:
            return ValueError('mean, max, or min are required')

        #return plt.show()

    #DRIEST/WETTEST - DONE
    def peak_cond(self):
        df['day'] = df[self.date_col].dt.day
        df['month'] = df[self.date_col].dt.month
        df['year'] = df[self.date_col].dt.month

        #highest humidity (smallest spread of temp and dewpoint, w/ the highest humidity reading)
        #high_vals = df.groupby(['day'], as_index=False)[[self.dew_point_col, self.relative_humidity_col, self.temp_col]].mean()
        df['mod_dp'] = df[self.temp_col] - df[self.dew_point_col]
        df['hum_combo'] = df['mod_dp']/df[self.relative_humidity_col]
        day_df = df.groupby(['year','day'], as_index=False)['hum_combo'].mean()
        day_df['year'] = day_df['year'].apply(self.mon_lbl)
        day_df['year'] = day_df['year'].astype(str)
        day_df['day'] = day_df['day'].apply(self.mon_lbl)
        day_df['day'] = day_df['day'].astype(str)
        day_df['date'] = day_df['year'] + day_df['day']
        day_df.plot(x='date',y='hum_combo',kind='line')
        #add marker for the driest and wettest conditions
        return plt.show()


    #COMFORT RATING - DONE
    def peak_comfort(self):
        df['day'] = df['time'].dt.day
        df['peak_cond'] = self.peak_cond()
        day_data_max = df.copy(deep=True)
        day_data_max['datecb'] = day_data_max['time'].dt.strftime('%m/%d/%Y')
        day_data_max = day_data_max[[self.temp_col,'time','hum_combo','datecb']]
        day_data_max = day_data_max.groupby(['datecb'],as_index=False)[self.temp_col,'hum_combo'].max()
        day_data_max = day_data_max.rename(columns={'hum_combo': 'driest'})
        day_data_max['datecb'] = pd.to_datetime(day_data_max['datecb'])
        day_data_max['day'] = day_data_max['datecb'].dt.day
        day_data_max['month'] = day_data_max['datecb'].dt.month
        day_data_max['year'] = day_data_max['datecb'].dt.year
        day_data_max = day_data_max.groupby(['month','day'],as_index=False)[self.temp_col,'driest'].mean()
        day_data_max['month'] = day_data_max['month'].apply(self.mon_lbl)
        day_data_max['month'] = day_data_max['month'].astype(str)
        day_data_max['day'] = day_data_max['day'].apply(self.mon_lbl)
        day_data_max['day'] = day_data_max['day'].astype(str)
        day_data_max['moday'] = day_data_max['month'] + day_data_max['day']
        day_data_max = day_data_max.drop(columns=['month','day'])
        #do the same for max and then merge the two 
        
        day_data_min = df.copy(deep=True)
        day_data_min['datecb'] = day_data_min['time'].dt.strftime('%m/%d/%Y')
        day_data_min = day_data_min[[self.temp_col,'time','hum_combo','datecb']]
        day_data_min = day_data_min.groupby(['datecb'],as_index=False)[self.temp_col,'hum_combo'].min()
        day_data_min = day_data_min.rename(columns={'hum_combo': 'wettest'})
        day_data_min['datecb'] = pd.to_datetime(day_data_min['datecb'])
        day_data_min['day'] = day_data_min['datecb'].dt.day
        day_data_min['month'] = day_data_min['datecb'].dt.month
        day_data_min['year'] = day_data_min['datecb'].dt.year
        day_data_min = day_data_min.groupby(['month','day'],as_index=False)[self.temp_col,'wettest'].mean()
        day_data_min['month'] = day_data_min['month'].apply(self.mon_lbl)
        day_data_min['month'] = day_data_min['month'].astype(str)
        day_data_min['day'] = day_data_min['day'].apply(self.mon_lbl)
        day_data_min['day'] = day_data_min['day'].astype(str)
        day_data_min['moday'] = day_data_min['month'] + day_data_min['day']
        day_data_min = day_data_min.drop(columns=['month','day'])

        mrg_comf = day_data_max.merge(day_data_min, how='left', on='moday')
        mrg_comf = mrg_comf.rename(columns={'temperature_2m_x':'max_temp','temperature_2m_y':'min_temp'})
        
        mrg_comf['max_dry'] = mrg_comf['driest'].max()
        mrg_comf['dry_dif'] = ((mrg_comf['driest']/mrg_comf['max_dry']) - 1) * -1
        mrg_comf['dry_score'] = mrg_comf['dry_dif'].apply(self.comf_scrng)
        #the higher the value in the 'wettest' column the drier it is
        mrg_comf['max_wet'] = mrg_comf['wettest'].max()
        mrg_comf['wet_dif'] = ((mrg_comf['wettest']/mrg_comf['max_wet']) - 1) * -1
        mrg_comf['wet_score'] = mrg_comf['wet_dif'].apply(self.comf_scrng)
        mrg_comf['tot_hum_scr'] = mrg_comf['dry_score'] + mrg_comf['wet_score']


        temp_df = mrg_comf.copy(deep=True)
        temp_df['high_dif'] = abs((temp_df['max_temp']/77) - 1)
        temp_df['htmp_score'] = temp_df['high_dif'].apply(self.comf_scrng)
        temp_df['low_dif'] = abs((temp_df['min_temp']/57) - 1)
        temp_df['ltmp_score']= temp_df['low_dif'].apply(self.comf_scrng)
        temp_df['temp_score'] = (temp_df['htmp_score']+temp_df['ltmp_score'])/2
        temp_df = temp_df[['moday','tot_hum_scr','temp_score']]
        temp_df['hum_adj'] = temp_df['tot_hum_scr'] * 0.02
        temp_df['comfort_score'] = temp_df['temp_score'] + temp_df['hum_adj']
        temp_df = temp_df[['moday','comfort_score']]
        temp_df.plot(y='comfort_score',x='moday')
        plt.title(f'{self.location} Discomfort Score')
    
        return plt.show()

        #parameters are temp and humidity
        #temp: highs from 75-80, lows from 55-60
        #max daily relh as low as possible, max dp vs temp diff in day 


if __name__=='__main__':
    h = humdity_calcs('time','relativehumidity_2m','dewpoint_2m','temperature_2m',location='Phoenix')
    h.peak_comfort()




