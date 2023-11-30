import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import threading
import time

from smart_home_env import SmartHomeTempControlEnv
from smart_home_config import CONFIG

global SIMULATION_THREAD_STARTED
SIMULATION_THREAD_STARTED = False

simulation = SmartHomeTempControlEnv(CONFIG)
simulation.reset()

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-graph', animate=True),
    dcc.Interval(
        id='graph-update',
        interval=1*1000,  # in milliseconds
        n_intervals=0
    ),
])

# Callback for updating the live graph
@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_graph_scatter(n):
    time, temp, heating, cooling = simulation.get_latest_data()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=temp, name='Temperature', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=time, y=heating, name='Heating Meter', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=time, y=cooling, name='Cooling Meter', mode='lines+markers'))

    fig.update_layout(
        xaxis=dict(
            type='date',
            title='Time'
        ),
        yaxis=dict(
            title='Temprature (Celsius)'
        ),
        title='Smart Home Temperature Control Simulation'
    )

    return fig

def run_simulation():
    global SIMULATION_THREAD_STARTED
    if not SIMULATION_THREAD_STARTED:
        SIMULATION_THREAD_STARTED = True
        for _ in range(10000):  # Run for 10000 steps
            action = simulation.action_space.sample()
            simulation.step(action)
            time.sleep(1)


if __name__ == '__main__':
    simulation_thread = threading.Thread(target=run_simulation)
    simulation_thread.start()
    app.run_server(debug=False, log_level='error')
