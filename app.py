import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import threading
import time

from smart_home_env import SmartHomeTempControlEnv
from smart_home_config import CONFIG

simulation = SmartHomeTempControlEnv(CONFIG)
simulation.reset()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Smart Home Temperature Control Simulation"),
    dcc.Graph(id='temperature-graph', animate=True),
    dcc.Graph(id='control-graph', animate=True),
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
        xaxis=dict(type='date', title='Time'),
        yaxis=dict(title='Temperature (°C)'),
        title='Temperature Over Time'
    )

    return fig

# Callback for updating the control graph
@app.callback(Output('control-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_control_graph(n):
    time_data, heating, cooling = simulation.get_control_data()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_data, y=heating, name='Heating Meter', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=time_data, y=cooling, name='Cooling Meter', mode='lines+markers'))

    fig.update_layout(
        xaxis=dict(type='date', title='Time'),
        yaxis=dict(title='Meter Reading'),
        title='Heating and Cooling Controls Over Time'
    )

    return fig

# Callback for updating the occupancy graph
@app.callback(Output('occupancy-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_occupancy_graph(n):
    time_data, occupancy = simulation.get_occupancy_data()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_data, y=occupancy, name='Occupancy', mode='lines+markers'))

    fig.update_layout(
        xaxis=dict(type='date', title='Time'),
        yaxis=dict(title='Occupancy'),
        title='Occupancy Over Time'
    )

    return fig

def run_simulation():
    for _ in range(10000):  # Run for 10000 steps
        action = simulation.action_space.sample()
        simulation.step(action)
        time.sleep(1)


if __name__ == '__main__':
    simulation_thread = threading.Thread(target=run_simulation)
    simulation_thread.start()
    app.run_server(debug=False)
