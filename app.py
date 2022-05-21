from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud

app = Dash(__name__)

continents = ['America', 'Europe', 'Africa', 'Asia']

attractions_ne = {
    'America': 'NewYork, Seatle, Boston',
    'Europe': 'Stuttgart, Lisabon, London, Berlin',
    'Africa': 'Kapstadt, Johannesburg, Nairobi',
    'Asia': 'Tokio, Shanghai, Wuhan'
}

# create HTML layout including all input and output components
app.layout = html.Div([
    html.Div([
        html.H1(children='Dashboard Datascience - Travel location finder'),
        # country filter for entire dashboard
        html.Div([
            html.Label('Continent'),
            dcc.Dropdown(
                continents,
                id='global_continent'
            )
        ])
    ]),
    # all four graphs in grid order
    html.Div([
        # section for first two graphs
        html.Div([
            dcc.Graph(
                id='wordcloud_ne',
            ),
            dcc.Graph(
                id='wordcloud_no_ne'
            )
        ], style={'display': 'flex', 'flex-direction': 'row'}),
        # section for second two graphs
        html.Div([
            dcc.Graph(
                id='attraction_combinations',
            ),
            dcc.Graph(
                id='individual_visualisation'
            )
        ], style={'display': 'flex', 'flex-direction': 'row'})
    ])
])
# callback for global continent filtering
@app.callback(
    Output(component_id='wordcloud_ne', component_property='figure'),
    Output(component_id='wordcloud_no_ne', component_property='figure'),
    #Output(component_id='attraction_combinations', component_property='figure'),
    #Output(component_id='individual_visualisation', component_property='figure'),
    Input(component_id='global_continent', component_property='value')
    )
def apply_global_filter(global_continent):
    if global_continent is not None:
        ne_img = WordCloud().generate(attractions_ne[global_continent])
        wordcloud_ne = px.imshow(ne_img)
        wordcloud_no_ne = px.imshow(ne_img)
    else:
        whole_str = str(x for x in attractions_ne.values())
        ne_img = WordCloud().generate(whole_str)
        wordcloud_ne = px.imshow(ne_img)
        wordcloud_no_ne = px.imshow(ne_img)
    return wordcloud_ne, wordcloud_no_ne

# wordcloud wit NE locations

# wordcloud with attractions (no NE)

# visualisation of top 10 tourist attraction combinations

# own visualisation

if __name__ == '__main__':
    app.run_server(debug=True)
