
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

    #APP WORKS
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

    #APP WORKS
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

    #IN DEV
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

    #IN DEV
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

#IN DEV
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
            ax.plot(pltdf)
            ax.set_title(f'{self.temp_type} Temps') 
            st.pyplot(fig)
    
    def date_temp_plot_day(self,dydf):
        if self.dataframe is None:
            raise ValueError("Dataframe not available. Call date_temps first.")
        else:
            fig, ax = plt.subplots()
            ax.plot(dydf['combo'],dydf['temperature_2m'])
            num_ticks = 12
            num_data_points = len(dydf['combo'])
            step = max(1, num_data_points // num_ticks)
            ax.set_xticks(dydf['combo'][::step])
            plt.xticks(rotation=45)
            ax.set_title(f'{self.temp_type} Temps') 
            st.pyplot(fig)

    #APP WORKING
    def date_temps(self):
        #UNDER DEV
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
            return temp_day

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
        elif self.date_type == 'year':
            #JUST USE MONTHLY FOR NOW
            temp_year = self.dataframe.groupby(self.dataframe['time'].dt.year)['temperature_2m'].mean()
            return temp_year
        else:
            return ValueError('day, month, or year required')

    #APP WORKING
    def stdv_temps(self):
        if self.temp_type == 'Mean':
            #Don't use mean for now as people don't really think about mean temps 
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
        #fin_std_df.plot(x='combo',y=['mean_temp','p1_stdp','p1_stdn'])
        fig, ax = plt.subplots()
        ax.plot(fin_std_df['combo'], fin_std_df['mean_temp'], label='mean_temp')
        ax.plot(fin_std_df['combo'], fin_std_df['p1_stdp'], label='p1_stdp')
        ax.plot(fin_std_df['combo'], fin_std_df['p1_stdn'], label='p1_stdn')
        num_ticks = 12
        num_data_points = len(fin_std_df['combo'])
        step = max(1, num_data_points // num_ticks)
        ax.set_xticks(fin_std_df['combo'][::step])
        plt.xticks(rotation=45)
        ax.set_title(f'{self.temp_type} Temp Standard Deviations') 
        st.pyplot(fig)

    #APP WORKING
    def diurnal_rng(self):
        #Need to figure out how these can return a plot and the dataframe for a different method like this
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
        fig, ax = plt.subplots()
        ax.plot(dr_merge['time'], dr_merge['diurnal'])
        ax.set_title('Diurnal Temps') 
        st.pyplot(fig)

#IN DEV
class precipitation_calcs:
    
    def __init__(self, dataframe, date_col, precip):
        self.dataframe = dataframe
        self.date_col = date_col
        self.precip  = precip 

    #APP WORKS
    def monthly_prec(self): 
        precip_day = self.dataframe.groupby(pd.Grouper(key = self.date_col, freq = 'M'))[self.precip].sum()
        prec_df = pd.DataFrame(precip_day)
        new_prec = prec_df.reset_index()
        sorted_prec = new_prec.groupby(new_prec[self.date_col].dt.month)[self.precip].mean()
        
        fig, ax = plt.subplots()
        ax.plot(sorted_prec)
        ax.set_title('Monthly Precipitation') 
        st.pyplot(fig)

    #APP WORKS...but some of the data seems wrong (i.e. it should not be more than 30)
    def morainy_cnt(self): #DONE - but need to validate what this does
        #could pass a different date grouping through the initialization function
        precip_cnt = self.dataframe.groupby(pd.Grouper(key = self.date_col, freq = 'M'))[self.precip].sum()
        precip_df = pd.DataFrame(precip_cnt)
        precip_df = precip_df.reset_index()
        precip_df['month'] = precip_df[self.date_col].dt.month
        precip_df = precip_df.groupby(['month'], as_index=False)[self.precip].sum()
        #precip_df['precipitation'] = precip_df['precipitation'].round()
        #precip_df.plot(x='month',y='precipitation')
        
        fig, ax = plt.subplots()
        ax.plot(precip_df['month'],precip_df['precipitation'])
        ax.set_title('Rainy Days per Month')
        st.pyplot(fig)

#APP WORKING
class humdity_calcs:
    
    def __init__(self, dataframe, date_col='na', relative_humidity_col='na', dew_point_col='na', temp_col='na', location='na'):
        self.dataframe = dataframe
        self.date_col = date_col
        self.relative_humidity_col = relative_humidity_col
        self.dew_point_col = dew_point_col
        self.temp_col = temp_col
        self.location = location

    #APP WORKS
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

    #APP WORKS
    def comf_scrng(self,row):
        return 10 + (10 * row)

    #APP WORKS
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

    #APP WORKS
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

        fig, ax = plt.subplots()
        ax.plot(temp_df['moday'], temp_df['comfort_score'])
        num_ticks = 12
        num_data_points = len(temp_df['moday'])
        step = max(1, num_data_points // num_ticks)
        ax.set_xticks(temp_df['moday'][::step])
        plt.xticks(rotation=45)
        ax.set_yticklabels([])
        ax.set_title('Discomfort Rating') 
        st.pyplot(fig)

        #parameters are temp and humidity
        #temp: highs from 75-80, lows from 55-60
        #max daily relh as low as possible, max dp vs temp diff in day 

#IN DEV
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
        map = st_folium(m, height=350, width=700)
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
            hum.peak_comfort()
            if data is not None:
                st.write('Discomfort determined by higher lines as displayed by the line plot.')
        elif analysis_type == 'Monthly Precipitation':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_precip()
            prec = precipitation_calcs(dataframez,date_col='time',precip='precipitation')
            prec.monthly_prec()
            if data is not None:
                st.write('Description of the monthly prec.')
        elif analysis_type == 'Rainy Days':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_precip()
            prec = precipitation_calcs(dataframez,date_col='time',precip='precipitation')
            prec.morainy_cnt()
        elif analysis_type == 'Monthly-Hi':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',date_type='month',temp_type='High')
            hi_df = tempz.date_temps()
            tempz.date_temp_plot_month(hi_df)
        elif analysis_type == 'Monthly-Lo':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',date_type='month',temp_type='Low')
            lo_df = tempz.date_temps()
            tempz.date_temp_plot_month(lo_df)
        elif analysis_type == 'Monthly-Mean':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',date_type='month',temp_type='Mean')
            mn_df = tempz.date_temps()
            tempz.date_temp_plot_month(mn_df)
        elif analysis_type == 'Daily-Mean':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',date_type='day',temp_type='Mean')
            mn_dy = tempz.date_temps()
            tempz.date_temp_plot_day(mn_dy)
        elif analysis_type == 'Daily-Hi':
            #Under Dev
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',date_type='day',temp_type='High')
            hi_dy = tempz.date_temps()
            tempz.date_temp_plot_day(hi_dy)
        elif analysis_type == 'Daily-Lo':
            #Under Dev
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',date_type='day',temp_type='Low')
            lo_dy = tempz.date_temps()
            tempz.date_temp_plot_day(lo_dy)
        elif analysis_type == 'Diurnal Range':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',temp_type='High')
            tempz.diurnal_rng()
        elif analysis_type == 'Hi-Standard Devs':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',temp_type='High')
            tempz.stdv_temps()
        elif analysis_type == 'Lo-Standard Devs':
            api = data_load(latitude=data[0],longitude=data[1])
            dataframez = api.data_processing_temp()
            tempz = temperature_calcs(dataframe=dataframez,date_col='time',temp_type='Low')
            tempz.stdv_temps()
        else:
            pass

if __name__=='__main__':
    main()





