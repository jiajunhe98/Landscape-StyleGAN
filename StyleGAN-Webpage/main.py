import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
from generate_pictures import *
import matplotlib.pyplot as plt


app = dash.Dash()

server = app.server


def get_initial_fig():
    img = plt.imread("init.jpg")
    fig = px.imshow(img)
    fig.update_layout(coloraxis_showscale=False, transition_duration=100, template="simple_white")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

all_options = {
    0: ['Random', 'Aurora', 'No Aurora'],
    1: ['Random', 'Dust or Dawn', 'Daytime'],
    4: []
}

app.layout = html.Div([
    html.H1(children="Taking Photos for Non-existing Landscapes", style={'textAlign': 'center'}),
    html.Br(),
    html.H6(children="Number of photos", style={'textAlign': 'center'}),
    dcc.Slider(
        id='num',
        min=2,
        max=8,
        value=4,
        marks={2:2, 4:4, 6:6, 8:8},
        step=2,
    ),
    html.H6(children="Choosing the time", style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='time',
        options=[
            {'label': 'Night', 'value': 0},
            {'label': 'Day', 'value': 1},
            {'label': 'Random Time', 'value': 4}
        ], value=4
    ),
    dcc.RadioItems(id='weather', style={'textAlign': 'center'}),
    html.H5(children="  ", style={'textAlign': 'center'}),
    dbc.Row([
        dbc.Col([
            dbc.Button(id='update-button', n_clicks=0, children='Shoot!', style={'textAlign': 'center'}),
        ], style={'textAlign': 'center'})
    ]),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=dcc.Graph(id='landscape', figure=get_initial_fig()),
    ),
    html.A("CHeck the source code here", href='https://github.com/hejj16/Landscape-StyleGAN', target="_blank")

], style={'width': '60%',
          'display': 'inline-block',
          'padding-left': '20%',
          'padding-right': '20%'})


@app.callback(Output('weather', 'options'), Input('time', 'value'))
def update_option(value):
    if value is not None:
        return [{'label': i, 'value': i} for i in all_options[value]]
    else:
        return []


@app.callback(
    Output('weather', 'value'),
    Input('weather', 'options'))
def set_cities_value(available_options):
    if available_options:
        return available_options[0]['value']
    else:
        return 4


@app.callback(Output('landscape', 'figure'),
            Input("update-button", "n_clicks"),
            [State('num', 'value'), State('time', 'value'), State('weather', 'value')]
            )
def update_figure_2(click, num, time, details):
    if click is not None and click != 0:
        if num <= 2:
            nrow = num
        else: nrow = num // 2

        if time == 0:
            detail = {'Random': 4, 'Aurora': 0, 'No Aurora': 1}[details]
        elif time == 1:
            detail = {'Random': 4, 'Dust or Dawn': 0, 'Daytime': 1}[details]
        else:
            detail = 4

        generate(num, nrow, time, detail)

        img = plt.imread("save.jpg")
        fig = px.imshow(img)

        fig.update_layout(coloraxis_showscale=False, transition_duration=100, template="simple_white")
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig
    else:
        return get_initial_fig()


if __name__ == '__main__':

    app.run_server()



