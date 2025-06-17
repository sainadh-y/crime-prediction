import pandas as pd
from prophet import Prophet
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

class CrimeProphetPredictor:
    def __init__(self, data_path='crime_rate_summary.csv'):
        self.data = pd.read_csv(data_path)
        self.models = {}
        self.prepare_data()

    def prepare_data(self):
        # Remove cluster column if exists
        if 'cluster' in self.data.columns:
            self.data = self.data.drop(columns=['cluster'])

        # Create a date column for Prophet
        self.data['ds'] = pd.to_datetime(self.data[['YEAR', 'MONTH']].assign(DAY=1))

        # Aggregate crime counts by NEIGHBOURHOOD, TYPE, and date
        self.agg_data = self.data.groupby(['NEIGHBOURHOOD', 'TYPE', 'ds'])['crime_rate'].sum().reset_index()

        # Get unique neighbourhoods and crime types
        self.neighbourhoods = self.agg_data['NEIGHBOURHOOD'].unique()
        self.crime_types = self.agg_data['TYPE'].unique()

    def train_models(self):
        # Train a Prophet model for each NEIGHBOURHOOD and TYPE with improved parameters
        for neigh in self.neighbourhoods:
            for crime in self.crime_types:
                df = self.agg_data[(self.agg_data['NEIGHBOURHOOD'] == neigh) & (self.agg_data['TYPE'] == crime)][['ds', 'crime_rate']]
                df = df.rename(columns={'crime_rate': 'y'})
                if len(df) < 2:
                    continue  # Not enough data to train
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.5,
                    seasonality_prior_scale=10.0
                )
                # Add monthly seasonality explicitly
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                model.fit(df)
                self.models[(neigh, crime)] = model

    def predict(self, neighbourhood, month, year, plot=False):
        # Predict crime counts for given neighbourhood, month, year
        date = pd.to_datetime({'year': [year], 'month': [month], 'day': [1]})
        preds = []
        total = 0

        for (neigh, crime), model in self.models.items():
            if neigh == neighbourhood:
                future = pd.DataFrame({'ds': date})
                forecast = model.predict(future)
                pred = max(0, int(round(forecast['yhat'].values[0])))
                if pred > 0:
                    preds.append((crime, pred))
                    total += pred
                    if plot:
                        print(f"Showing visualizations for {crime} in {neigh} for {month}/{year}...")
                        self.plot_forecast(model, crime, neigh)
                        self.plot_components(model)
                        self.plot_crime_type_distribution(neigh, month, year)
                        self.plot_heatmap(neigh)
                        self.plot_trend_comparison(neigh)
                        self.plot_forecast_with_uncertainty(model, crime, neigh)

        return {
            'predicted_crimes': preds,
            'total_crimes': total
        }

    def plot_forecast(self, model, crime_type, neighbourhood):
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=model.history['ds'], y=model.history['y'], mode='markers', name='Historical'))
        fig.update_layout(title=f'Forecast for {crime_type} in {neighbourhood}', xaxis_title='Date', yaxis_title='Crime Rate')
        fig.show()

    def plot_components(self, model):
        # Generate forecast dataframe for components plot
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        fig = model.plot_components(forecast)
        fig.show()

    def plot_crime_type_distribution(self, neighbourhood, month, year):
        # Pie chart of crime type distribution for a single month
        df = self.data[(self.data['NEIGHBOURHOOD'] == neighbourhood) & (self.data['YEAR'] == year) & (self.data['MONTH'] == month)]
        if df.empty:
            print(f"No data for {neighbourhood} in {month}/{year} to plot crime type distribution.")
            return
        crime_counts = df.groupby('TYPE')['crime_rate'].sum()
        plt.figure(figsize=(8, 8))
        plt.pie(crime_counts, labels=crime_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title(f'Crime Type Distribution in {neighbourhood} for {month}/{year}')
        plt.show()

    def plot_heatmap(self, neighbourhood):
        # Heatmap of crime intensity by month and year
        df = self.data[self.data['NEIGHBOURHOOD'] == neighbourhood]
        if df.empty:
            print(f"No data for {neighbourhood} to plot heatmap.")
            return
        pivot = df.pivot_table(index='YEAR', columns='MONTH', values='crime_rate', aggfunc='sum', fill_value=0)
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt='g', cmap='YlOrRd')
        plt.title(f'Crime Intensity Heatmap by Month and Year in {neighbourhood}')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.show()

    def plot_trend_comparison(self, neighbourhood):
        # Line chart comparing crime trends by type over time
        df = self.data[self.data['NEIGHBOURHOOD'] == neighbourhood]
        if df.empty:
            print(f"No data for {neighbourhood} to plot trend comparison.")
            return
        plt.figure(figsize=(14, 7))
        for crime_type in self.crime_types:
            crime_data = df[df['TYPE'] == crime_type].groupby(['YEAR', 'MONTH'])['crime_rate'].sum().reset_index()
            crime_data['date'] = pd.to_datetime(crime_data[['YEAR', 'MONTH']].assign(DAY=1))
            plt.plot(crime_data['date'], crime_data['crime_rate'], label=crime_type)
        plt.title(f'Crime Trend Comparison by Type in {neighbourhood}')
        plt.xlabel('Date')
        plt.ylabel('Crime Rate')
        plt.legend()
        plt.show()

    def plot_forecast_with_uncertainty(self, model, crime_type, neighbourhood):
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        plt.figure(figsize=(10, 6))
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.3, label='Uncertainty Interval')
        plt.scatter(model.history['ds'], model.history['y'], color='red', label='Historical')
        plt.title(f'Time Series Forecast with Uncertainty Intervals for {crime_type} in {neighbourhood}')
        plt.xlabel('Date')
        plt.ylabel('Crime Rate')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    neighbourhood = input("Enter the neighbourhood: ")
    month = int(input("Enter the month (1-12): "))
    year = int(input("Enter the year (e.g., 2023): "))
    plot = 'yes'

    predictor = CrimeProphetPredictor()
    predictor.train_models()
    result = predictor.predict(neighbourhood, month, year, plot=plot)

    if not result['predicted_crimes']:
        print(f"No predictions available for neighbourhood {neighbourhood} in {month}/{year}.")
    else:
        print(f"Predicted crimes for {neighbourhood} in {month}/{year}:")
        for crime_type, count in result['predicted_crimes']:
            print(f"  {crime_type}: {count}")
        print(f"Total predicted crimes: {result['total_crimes']}")
