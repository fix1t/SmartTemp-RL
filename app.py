import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import threading
import time

from smart_home_env import SmartHomeTempControlEnv

simulation = SmartHomeTempControlEnv()
simulation.reset()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Smart Home Temperature Control Simulation"),
    dcc.Graph(id='temperature-graph', animate=True),
    dcc.Graph(id='heating-system-graph', animate=True),
    dcc.Graph(id='occupancy-graph', animate=True),
    dcc.Interval(
        id='graph-update',
        interval=1*1000,  # in milliseconds
        n_intervals=0
    ),
])

# Callback for updating the temperature graph
@app.callback(Output('temperature-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_temperature_graph(n):
    time_data, temp, outside_temp = simulation.get_temperature_data()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_data, y=temp, name='Indoor Temperature', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=time_data, y=outside_temp, name='Outside Temperature', mode='lines+markers'))

    fig.update_layout(
        xaxis=dict(type='date', title='Time', autorange=True),
        yaxis=dict(title='Temperature (°C)', autorange=True),
        title='Temperature Over Time',
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig

# Callback for updating the control graph
@app.callback(Output('heating-system-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_heating_system_graph(n):
    time_data, heating = simulation.get_heating_data()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_data, y=heating, name='Heating Meter', mode='lines+markers'))

    fig.update_layout(
        xaxis=dict(type='date', title='Time', autorange=True),
        yaxis=dict(title='Meter Reading', autorange=True),
        title='Heating Over Time',
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig

# Callback for updating the occupancy graph
@app.callback(Output('occupancy-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_occupancy_graph(n):
    time_data, people_presence_data = simulation.get_occupancy_data()

    fig = go.Figure()

    # Iterate through each person and add traces to the figure
    for person, presence_data in people_presence_data.items():
        fig.add_trace(go.Scatter(x=time_data, y=presence_data, name=person, mode='lines+markers'))

    fig.update_layout(
        xaxis=dict(type='date', title='Time', autorange=True),
        yaxis=dict(title='At home', autorange=True),
        title='People At Home Over Time',
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig

def run_simulation():
    for _ in range(4 * 24 * 30):
        action = simulation.action_space.sample()
        simulation.step(action)
        time.sleep(0.01)


if __name__ == '__main__':
    simulation_thread = threading.Thread(target=run_simulation)
    simulation_thread.start()
    app.run_server(debug=False)
