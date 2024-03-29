import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import threading
import time

class SimulationRenderer:
    def __init__(self, simulation):
        self.simulation = simulation
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Smart Home Temperature Control Simulation"),
            dcc.Graph(id='score-graph', animate=True),
            dcc.Graph(id='temperature-graph', animate=True),
            dcc.Graph(id='heating-system-graph', animate=True),
            dcc.Graph(id='occupancy-graph', animate=True),
            dcc.Interval(
                id='graph-update',
                interval=1*2000,
                n_intervals=0
            ),
        ])

    def setup_callbacks(self):
        # Score graph
        @self.app.callback(Output('score-graph', 'figure'),
                           [Input('graph-update', 'n_intervals')])
        def update_score_graph(n):
            time_data, rewards = self.simulation.get_reward_data()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_data, y=rewards, mode='lines+markers'))

            fig.update_layout(
                xaxis=dict(type='date', title='Time', autorange=True),
                yaxis=dict(title='Score', autorange=True),
                title='Score Over Time',
                margin=dict(l=40, r=40, t=40, b=40),
            )

            return fig

        # Temperature graph
        @self.app.callback(Output('temperature-graph', 'figure'),
                           [Input('graph-update', 'n_intervals')])
        def update_temperature_graph(n):
            time_data, temp, outside_temp = self.simulation.get_temperature_data()

            # Calculate the target temperature trace
            target_temp_trace = [self.simulation.target_temperature] * len(time_data)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_data, y=temp, name='Indoor', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=time_data, y=outside_temp, name='Outdoor', mode='lines+markers'))
            # Add the target temperature trace
            fig.add_trace(go.Scatter(x=time_data, y=target_temp_trace, name='Target', mode='lines', line=dict(color='green', dash='dash')))

            fig.update_layout(
                xaxis=dict(type='date', title='Time', autorange=True),
                yaxis=dict(title='Temperature (°C)', autorange=True),
                title='Temperature Over Time',
                margin=dict(l=40, r=40, t=40, b=40),
            )

            return fig


        # Heating system graph
        @self.app.callback(Output('heating-system-graph', 'figure'),
                           [Input('graph-update', 'n_intervals')])
        def update_heating_system_graph(n):
            time_data, heating = self.simulation.get_heating_data()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_data, y=heating, name='Heating Meter', mode='lines+markers'))

            fig.update_layout(
                xaxis=dict(type='date', title='Time', autorange=True),
                yaxis=dict(title='Meter Reading', autorange=True),
                title='Heating Over Time',
                margin=dict(l=40, r=40, t=40, b=40),
            )

            return fig

        # Occupancy graph
        @self.app.callback(Output('occupancy-graph', 'figure'),
                           [Input('graph-update', 'n_intervals')])
        def update_occupancy_graph(n):
            time_data, people_presence_data = self.simulation.get_occupancy_data()

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

    def run_random_simulation(self):
        for _ in range(4 * 24 * 30):
            action = self.simulation.action_space.sample()
            self.simulation.step(action)
            time.sleep(0.01)

    def run_server(self):
        self.thread = threading.Thread(target=lambda: self.app.run_server(debug=False), daemon=True)
        self.thread.start()


