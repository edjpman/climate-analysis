

import streamlit as st

import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import streamlit as st 
from streamlit_folium import st_folium, folium_static
import folium
import pydeck as pdk
import numpy as np

app_title = 'Climate Analysis App'
app_subheader = 'Interactive Map'


class data_load:
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    #Data Processing for Humidity Methods
    def data_processing_humidity(self):
        # Sets up the initial start date and end date for the first 5-year window
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365*5)).strftime('%Y-%m-%d')

        # Construct the URL with the initial start_date and end_date
        url = f'https://archive-api.open-meteo.com/v1/archive?latitude={self.latitude}&longitude={self.longitude}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m&hourly=dewpoint_2m&hourly=relativehumidity_2m'

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
        return df

    #Data Processing for Precipitation Methods
    def data_processing_precip(self):
        # Set up the initial start date and end date for the first 5-year window
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365*5)).strftime('%Y-%m-%d')

        # Construct the URL with the initial start_date and end_date
        url = f'https://archive-api.open-meteo.com/v1/archive?latitude={self.latitude}&longitude={self.longitude}&start_date={start_date}&end_date={end_date}&hourly=precipitation'

        # Retrieve the data and create a pandas DataFrame
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data).transpose()
        df = df[['time','precipitation']]
        df = df.transpose()
        df = df[['hourly']].transpose()
        df = df[['precipitation', 'time']].explode(['precipitation', 'time'])
        df['time'] = pd.to_datetime(df['time'])
        df['precipitation'] = (df['precipitation']/25.4)
        return df

    #Data Processing for Temperature Methods
    def data_processing_temp(self):
        # Set up the initial start date and end date for the first 5-year window
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365*5)).strftime('%Y-%m-%d')

        # Construct the URL with the initial start_date and end_date
        url = f'https://archive-api.open-meteo.com/v1/archive?latitude={self.latitude}&longitude={self.longitude}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m&hourly=apparent_temperature'

        # Retrieve the data and create a pandas DataFrame
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data).transpose()
        df = df[['time','temperature_2m','apparent_temperature']]
        df = df.transpose()
        df = df[['hourly']].transpose()
        df = df[['temperature_2m','apparent_temperature','time']].explode(['temperature_2m','apparent_temperature','time'])
        df['temperature_2m'] = (df['temperature_2m'] * 1.8) + 32
        df['time'] = pd.to_datetime(df['time'])
        return df

    #Data Processing for Wind Methods
    def data_processing_wind(self):
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365*5)).strftime('%Y-%m-%d')

        # Construct the URL with the initial start_date and end_date
        url = f'https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=winddirection_10m&hourly=windspeed_10m'

        # Retrieve the data and create a pandas DataFrame
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data).transpose()
        df = df[['time','winddirection_10m','windspeed_10m']]
        df = df.transpose()
        df = df[['hourly']].transpose()
        df = df[['winddirection_10m','windspeed_10m','time']].explode(['winddirection_10m','windspeed_10m','time'])
        df['time'] = pd.to_datetime(df['time'])

class temperature_calcs:
    
    def __init__(self, dataframe='na',date_col='na',date_type='na',temp_type='na'):
        self.dataframe = dataframe
        self.date_col = date_col
        self.date_type = date_type
        self.temp_type = temp_type

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

    def date_temp_plot_month(self,pltdf):
        if self.dataframe is None:
            raise ValueError("Dataframe not available. Call date_temps first.")
        else:
            fig, ax = plt.subplots()
            fig.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            ax.plot(pltdf)
            ax.tick_params(axis='both', colors='#0E1117')
            ax.tick_params(axis='x', labelcolor='white')
            ax.tick_params(axis='y', labelcolor='white')

            ax.set_title(f'{self.temp_type} Temps', color='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)  
            st.pyplot(fig)
    
    def date_temp_plot_day(self,dydf):
        if self.dataframe is None:
            raise ValueError("Dataframe not available. Call date_temps first.")
        else:
            fig, ax = plt.subplots()
            fig.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            ax.plot(dydf['combo'],dydf['temperature_2m'])
            num_ticks = 12
            num_data_points = len(dydf['combo'])
            step = max(1, num_data_points // num_ticks)
            ax.tick_params(axis='both', colors='#0E1117')

            ax.set_xticks(dydf.index[::step]) 
            ax.set_xticklabels(dydf['combo'][::step], rotation=45, color='white')
            #ax.set_yticklabels([])
            ax.tick_params(axis='y', labelcolor='white')
        
            ax.set_title(f'{self.temp_type} Temps', color='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False) 
            st.pyplot(fig)

    def stdv_plot(self,stdf):
        fig, ax = plt.subplots()
        fig.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        ax.plot(stdf['combo'], stdf['mean_temp'], label='mean_temp')
        ax.plot(stdf['combo'], stdf['p1_stdp'], label='p1_stdp')
        ax.plot(stdf['combo'], stdf['p1_stdn'], label='p1_stdn')
        num_ticks = 12
        num_data_points = len(stdf['combo'])
        step = max(1, num_data_points // num_ticks)
        ax.tick_params(axis='both', colors='#0E1117')
        ax.set_xticks(stdf['combo'][::step])
        plt.xticks(rotation=45)
        ax.set_title(f'{self.temp_type} Temp Standard Deviations',color='white') 
        ax.tick_params(axis='x', labelcolor='white')
        ax.tick_params(axis='y', labelcolor='white')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        st.pyplot(fig)

    def diurnal_plot(self,drdf):
        fig, ax = plt.subplots()
        fig.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        ax.plot(drdf['time'], drdf['diurnal'])
        ax.tick_params(axis='both', colors='#0E1117')
        ax.set_title('Diurnal Temps',color='white')
        ax.tick_params(axis='x', labelcolor='white')
        ax.tick_params(axis='y', labelcolor='white')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False) 
        st.pyplot(fig)

    def date_temps(self):
        if self.date_type == 'day':
            #need to make sure this returns the average over the days
            if self.temp_type == 'Mean':  
                temp_day = self.dataframe.groupby(pd.Grouper(key = self.date_col, freq = 'D'))['temperature_2m'].mean()
                temp_day = temp_day.reset_index()
                temp_day['time'] = pd.to_datetime(temp_day['time'])
            elif self.temp_type == 'High':
                temp_day = self.dataframe.groupby(pd.Grouper(key = self.date_col, freq = 'D'))['temperature_2m'].max()
                temp_day = temp_day.reset_index()
                temp_day['time'] = pd.to_datetime(temp_day['time'])
            elif self.temp_type == 'Low':
                temp_day = self.dataframe.groupby(pd.Grouper(key = self.date_col, freq = 'D'))['temperature_2m'].min()
                temp_day = temp_day.reset_index()
                temp_day['time'] = pd.to_datetime(temp_day['time'])
            else:
                return ValueError('Mean, High, or Low required')
            temp_day = temp_day.reset_index()
            temp_day['month'] = temp_day[self.date_col].dt.month
            temp_day['month'] = temp_day['month'].apply(self.mon_lbl)
            temp_day['month'] = temp_day['month'].astype(str)
            temp_day['day'] = temp_day[self.date_col].dt.day
            temp_day['day'] = temp_day['day'].apply(self.mon_lbl)
            temp_day['day'] = temp_day['day'].astype(str)
            temp_day['combo'] = temp_day['month'] + temp_day['day']
            temp_day = temp_day.drop(columns=['month','day'])
            temp_day = temp_day.groupby(['combo'],as_index=False)['temperature_2m'].mean()
            day_tmpdf = temp_day.copy(deep=True)
            return day_tmpdf

        elif self.date_type == 'month':
            if self.temp_type == 'High':
                temp_month = self.dataframe.groupby(self.dataframe['time'].dt.date)['temperature_2m'].max()
                temp_month = temp_month.reset_index()
                temp_month['time'] = pd.to_datetime(temp_month['time'])
                temp_month = temp_month.groupby(temp_month['time'].dt.month)['temperature_2m'].mean()
                return temp_month
            elif self.temp_type == 'Mean':
                temp_month = self.dataframe.groupby(self.dataframe['time'].dt.month)['temperature_2m'].mean()
                return temp_month
            elif self.temp_type == 'Low':
                temp_month = self.dataframe.groupby(self.dataframe['time'].dt.date)['temperature_2m'].min()
                temp_month = temp_month.reset_index()
                temp_month['time'] = pd.to_datetime(temp_month['time'])
                temp_month = temp_month.groupby(temp_month['time'].dt.month)['temperature_2m'].mean()
                return temp_month
            else:
                ValueError('high, mean, or low required')
        else:
            return ValueError('day or month required')

    def stdv_temps(self):
        if self.temp_type == 'Mean':
            temp_month = self.dataframe.groupby(pd.Grouper(key = self.date_col, freq = 'D'))['temperature_2m'].mean()
        elif self.temp_type == 'High':
            temp_month = self.dataframe.groupby(pd.Grouper(key= self.date_col, freq= 'D'))['temperature_2m'].max()
        elif self.temp_type == 'Low':
            temp_month = self.dataframe.groupby(pd.Grouper(key= self.date_col, freq= 'D'))['temperature_2m'].min()
        else:
            ValueError('Mean, High, or Low required')

        temp_month = temp_month.reset_index()
        temp_month['month'] = temp_month[self.date_col].dt.month
        temp_month['month'] = temp_month['month'].apply(self.mon_lbl)
        temp_month['month'] = temp_month['month'].astype(str)
        temp_month['day'] = temp_month[self.date_col].dt.day
        temp_month['day'] = temp_month['day'].apply(self.mon_lbl)
        temp_month['day'] = temp_month['day'].astype(str)
        temp_month['combo'] = temp_month['month'] + temp_month['day']
        temp_month = temp_month.drop(columns=['month','day'])
        
        new_temp_month = temp_month.copy(deep=True)
        new_temp_month = new_temp_month.rename(columns={'temperature_2m':'hist_temp'})
        temp_scn = temp_month.copy(deep=True)
        temp_scn = temp_scn.groupby(['combo'],as_index=False)['temperature_2m'].mean()
        temp_scn = temp_scn.rename(columns={'temperature_2m':'mean_temp'})

        merged_temp = new_temp_month.merge(temp_scn, on='combo',how='left')
        merged_temp['squared_dif'] = (merged_temp['hist_temp'] - merged_temp['mean_temp'])**2
        final_year = merged_temp['time'].iloc[-1].year
        first_year = merged_temp['time'].iloc[0].year
        timeframe = final_year - first_year

        ssdif = merged_temp.copy(deep=True)
        ssdif = ssdif.groupby(['combo'],as_index=False)['squared_dif'].sum()
        ssdif['squared_dif'] = ssdif['squared_dif']/timeframe
        ssdif['squared_dif'] = ssdif['squared_dif'].fillna(0) #new test line
        ssdif['squared_dif'] = ssdif['squared_dif'].apply(np.sqrt)
        ssdif = ssdif.rename(columns={'squared_dif':'std_dev'})

        fin_std_df = temp_scn.copy(deep=True)
        fin_std_df = fin_std_df.merge(ssdif,on='combo',how='left')
        fin_std_df['p1_stdp'] = fin_std_df['mean_temp'] + (fin_std_df['std_dev']*3)
        fin_std_df['p1_stdn'] = fin_std_df['mean_temp'] + (fin_std_df['std_dev']*-3)
        return fin_std_df

    def diurnal_rng(self):
        ht = temperature_calcs(dataframe=self.dataframe,date_col='time',date_type='month',temp_type='High')
        lt = temperature_calcs(dataframe=self.dataframe,date_col='time',date_type='month',temp_type='Low')
        dr_high = ht.date_temps()
        dr_high = dr_high.reset_index()
        dr_low = lt.date_temps()
        dr_low = dr_low.reset_index()
        dr_merge = dr_high.merge(dr_low,on='time',how='left')
        dr_merge = dr_merge.rename(columns={'temperature_2m_x':'max_tmp','temperature_2m_y':'min_tmp'})
        dr_merge['diurnal'] = dr_merge['max_tmp'] - dr_merge['min_tmp']
        dr_merge = dr_merge[['time','diurnal']]
        return dr_merge

class precipitation_calcs:
    
    def __init__(self, dataframe, date_col, precip):
        self.dataframe = dataframe
        self.date_col = date_col
        self.precip  = precip 

    def mon_prec_plot(self,mpdf):
        fig, ax = plt.subplots()
        fig.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        ax.plot(mpdf)
        ax.tick_params(axis='both', colors='#0E1117')
        ax.set_title('Monthly Precipitation', color='white')
        ax.tick_params(axis='x', labelcolor='white')
        ax.tick_params(axis='y', labelcolor='white') 

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        st.pyplot(fig)

    def mo_rain_plot(self,mrdf):
        fig, ax = plt.subplots()
        fig.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        ax.plot(mrdf['month'],mrdf['precipitation'])
        ax.tick_params(axis='both', colors='#0E1117')
        ax.set_title('Rainy Days per Month', color='white')
        ax.tick_params(axis='x', labelcolor='white')
        ax.tick_params(axis='y', labelcolor='white') 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        st.pyplot(fig)

    def monthly_prec(self): 
        precip_day = self.dataframe.groupby(pd.Grouper(key = self.date_col, freq = 'M'))[self.precip].sum()
        prec_df = pd.DataFrame(precip_day)
        new_prec = prec_df.reset_index()
        sorted_prec = new_prec.groupby(new_prec[self.date_col].dt.month)[self.precip].mean()
        return sorted_prec

    def morainy_cnt(self): 
        precip_cnt = self.dataframe.groupby(pd.Grouper(key = self.date_col, freq = 'M'))[self.precip].sum()
        precip_df = pd.DataFrame(precip_cnt)
        precip_df = precip_df.reset_index()
        precip_df['month'] = precip_df[self.date_col].dt.month
        precip_df = precip_df.groupby(['month'], as_index=False)[self.precip].sum()
        return precip_df

class humdity_calcs:
    
    def __init__(self, dataframe, date_col='na', relative_humidity_col='na', dew_point_col='na', temp_col='na', location='na'):
        self.dataframe = dataframe
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

    def peak_plot(self,pddf):
        fig, ax = plt.subplots()
        fig.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117') 
        ax.plot(pddf['moday'], pddf['comfort_score'])
        num_ticks = 12
        num_data_points = len(pddf['moday'])
        step = max(1, num_data_points // num_ticks)
        ax.tick_params(axis='both', colors='#0E1117') 
        ax.set_xticks(pddf.index[::step])  
        ax.set_xticklabels(pddf['moday'][::step], rotation=45, color='white')
        ax.set_yticklabels([])
        ax.set_title('Discomfort Rating', color='white')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        st.pyplot(fig)

    def peak_cond(self):
        self.dataframe['day'] = self.dataframe[self.date_col].dt.day
        self.dataframe['month'] = self.dataframe[self.date_col].dt.month
        self.dataframe['year'] = self.dataframe[self.date_col].dt.month
        self.dataframe['mod_dp'] = self.dataframe[self.temp_col] - self.dataframe[self.dew_point_col]
        self.dataframe['hum_combo'] = self.dataframe['mod_dp']/self.dataframe[self.relative_humidity_col]
        day_df = self.dataframe.groupby(['year','day'], as_index=False)['hum_combo'].mean()
        day_df['year'] = day_df['year'].apply(self.mon_lbl)
        day_df['year'] = day_df['year'].astype(str)
        day_df['day'] = day_df['day'].apply(self.mon_lbl)
        day_df['day'] = day_df['day'].astype(str)
        day_df['date'] = day_df['year'] + day_df['day']
        day_df = day_df.drop(columns=['year','day'])

    def peak_comfort(self):
        self.dataframe['day'] = self.dataframe['time'].dt.day
        self.dataframe['peak_cond'] = self.peak_cond()
        day_data_max = self.dataframe.copy(deep=True)
        day_data_max['datecb'] = day_data_max['time'].dt.strftime('%m/%d/%Y')
        day_data_max = day_data_max[[self.temp_col,'time','hum_combo','datecb']]
        day_data_max = day_data_max.groupby(['datecb'],as_index=False)[[self.temp_col,'hum_combo']].max()
        day_data_max = day_data_max.rename(columns={'hum_combo': 'driest'})
        day_data_max['datecb'] = pd.to_datetime(day_data_max['datecb'])
        day_data_max['day'] = day_data_max['datecb'].dt.day
        day_data_max['month'] = day_data_max['datecb'].dt.month
        day_data_max['year'] = day_data_max['datecb'].dt.year
        day_data_max = day_data_max.groupby(['month','day'],as_index=False)[[self.temp_col,'driest']].mean()
        day_data_max['month'] = day_data_max['month'].apply(self.mon_lbl)
        day_data_max['month'] = day_data_max['month'].astype(str)
        day_data_max['day'] = day_data_max['day'].apply(self.mon_lbl)
        day_data_max['day'] = day_data_max['day'].astype(str)
        day_data_max['moday'] = day_data_max['month'] + day_data_max['day']
        day_data_max = day_data_max.drop(columns=['month','day'])
        
        day_data_min = self.dataframe.copy(deep=True)
        day_data_min['datecb'] = day_data_min['time'].dt.strftime('%m/%d/%Y')
        day_data_min = day_data_min[[self.temp_col,'time','hum_combo','datecb']]
        day_data_min = day_data_min.groupby(['datecb'],as_index=False)[[self.temp_col,'hum_combo']].min()
        day_data_min = day_data_min.rename(columns={'hum_combo': 'wettest'})
        day_data_min['datecb'] = pd.to_datetime(day_data_min['datecb'])
        day_data_min['day'] = day_data_min['datecb'].dt.day
        day_data_min['month'] = day_data_min['datecb'].dt.month
        day_data_min['year'] = day_data_min['datecb'].dt.year
        day_data_min = day_data_min.groupby(['month','day'],as_index=False)[[self.temp_col,'wettest']].mean()
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
        return temp_df

def main():
    st.set_page_config(layout="wide")
    st.title(app_title)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(app_subheader)
        st.write('Click to choose a coordinate on the map for the analysis')
        def get_pos(lat, lng):
            return lat, lng

        m = folium.Map()
        m.add_child(folium.LatLngPopup())
        map = st_folium(m, height=500, width=700)
        data = None
        if map.get("last_clicked"):
            data = get_pos(map["last_clicked"]["lat"], map["last_clicked"]["lng"])
    
    with col2:
        st.subheader('Analysis Type')
        analysis_type = st.selectbox('Select the Type of Analysis'
                     ,('Discomfort Scoring','Monthly Precipitation','Rainy Days','Monthly-Hi','Monthly-Lo','Monthly-Mean','Daily-Mean','Daily-Hi','Daily-Lo','Diurnal Range',
                       'Hi-Standard Devs','Lo-Standard Devs'))
        
        if analysis_type == 'Discomfort Scoring':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_humidity()
            hum = humdity_calcs(dataframez,date_col='time',temp_col='temperature_2m',relative_humidity_col='relativehumidity_2m',dew_point_col='dewpoint_2m')
            pcdf = hum.peak_comfort()
            hum.peak_plot(pddf=pcdf)
            csv_string_pc = pcdf.to_csv()
            st.download_button(label='Download CSV',data=csv_string_pc.encode('utf-8'),key='csv_file',file_name='peak_comfort.csv')
            if data is not None:
                st.write('Discomfort determined by higher lines as displayed by the line plot.')
        elif analysis_type == 'Monthly Precipitation':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_precip()
            prec = precipitation_calcs(dataframez,date_col='time',precip='precipitation')
            mopr = prec.monthly_prec()
            prec.mon_prec_plot(mopr)
            csv_string_mp = mopr.to_csv()
            st.download_button(label='Download CSV',data=csv_string_mp.encode('utf-8'),key='csv_file',file_name='monthly_prec.csv')
            if data is not None:
                st.write('Description of the monthly prec.')
        elif analysis_type == 'Rainy Days':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_precip()
            prec = precipitation_calcs(dataframez,date_col='time',precip='precipitation')
            rndy = prec.morainy_cnt()
            prec.mo_rain_plot(rndy)
            csv_string_rd = rndy.to_csv()
            st.download_button(label='Download CSV',data=csv_string_rd.encode('utf-8'),key='csv_file',file_name='rainy_days.csv')
        elif analysis_type == 'Monthly-Hi':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',date_type='month',temp_type='High')
            hi_df = tempz.date_temps()
            tempz.date_temp_plot_month(hi_df)
            csv_string_mh = hi_df.to_csv()
            st.download_button(label='Download CSV',data=csv_string_mh.encode('utf-8'),key='csv_file',file_name='monthly_hi.csv')
        elif analysis_type == 'Monthly-Lo':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',date_type='month',temp_type='Low')
            lo_df = tempz.date_temps()
            tempz.date_temp_plot_month(lo_df)
            csv_string_ml = lo_df.to_csv()
            st.download_button(label='Download CSV',data=csv_string_ml.encode('utf-8'),key='csv_file',file_name='monthly_lo.csv')
        elif analysis_type == 'Monthly-Mean':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',date_type='month',temp_type='Mean')
            mn_df = tempz.date_temps()
            tempz.date_temp_plot_month(mn_df)
            csv_string_mm = mn_df.to_csv()
            st.download_button(label='Download CSV',data=csv_string_mm.encode('utf-8'),key='csv_file',file_name='monthly_mn.csv')
        elif analysis_type == 'Daily-Mean':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',date_type='day',temp_type='Mean')
            mn_dy = tempz.date_temps()
            tempz.date_temp_plot_day(mn_dy)
            csv_string_dm = mn_dy.to_csv()
            st.download_button(label='Download CSV',data=csv_string_dm.encode('utf-8'),key='csv_file',file_name='daily_mn.csv')
        elif analysis_type == 'Daily-Hi':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',date_type='day',temp_type='High')
            hi_dy = tempz.date_temps()
            tempz.date_temp_plot_day(hi_dy)
            csv_string_dh = hi_dy.to_csv()
            st.download_button(label='Download CSV',data=csv_string_dh.encode('utf-8'),key='csv_file',file_name='daily_hi.csv')
        elif analysis_type == 'Daily-Lo':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',date_type='day',temp_type='Low')
            lo_dy = tempz.date_temps()
            tempz.date_temp_plot_day(lo_dy)
            csv_string_dl = lo_dy.to_csv()
            st.download_button(label='Download CSV',data=csv_string_dl.encode('utf-8'),key='csv_file',file_name='daily_lo.csv')
        elif analysis_type == 'Diurnal Range':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',temp_type='High')
            dr_df = tempz.diurnal_rng()
            tempz.diurnal_plot(dr_df)
            csv_string_dr = dr_df.to_csv()
            st.download_button(label='Download CSV',data=csv_string_dr.encode('utf-8'),key='csv_file',file_name='diurnal_rng.csv')
        elif analysis_type == 'Hi-Standard Devs':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',temp_type='High')
            standev_df_hi = tempz.stdv_temps()
            tempz.stdv_plot(standev_df_hi)
            csv_string_stdv_hi = standev_df_hi.to_csv()
            st.download_button(label='Download CSV',data=csv_string_stdv_hi.encode('utf-8'),key='csv_file',file_name='stdv_hi.csv')
        elif analysis_type == 'Lo-Standard Devs':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',temp_type='Low')
            standev_df_lo = tempz.stdv_temps()
            tempz.stdv_plot(standev_df_lo)
            csv_string_stdv_lo = standev_df_lo.to_csv()
            st.download_button(label='Download CSV',data=csv_string_stdv_lo.encode('utf-8'),key='csv_file',file_name='stdv_lo.csv')
        else:

            pass

if __name__=='__main__':
    main()

