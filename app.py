from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud

app = Dash(__name__)

# read csv files to dataframes
df_prep = pd.read_csv('data/results_df_prep.csv')
df_comb = pd.read_csv('data/results_top_combinations.csv')

# create list with countries for dropdown filter
countries = df_prep['Country'].unique()

# create HTML layout including all input and output components
app.layout = html.Div([
    html.Div([
        html.H1(children='Dashboard Data Science - Travel Location Finder'),
        # country filter for entire dashboard
        html.Div([
            html.Label('Countries'),
            dcc.Dropdown(
                countries,
                id='global_countries'
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
    Input(component_id='global_countries', component_property='value')
    )
def apply_global_filter(global_country):
    if global_country is not None:
        ne = df_prep.loc[df_prep['Country'] == global_country]['named_entities_spacy_small_plain'].to_string()
        no_ne = df_prep.loc[df_prep['Country'] == global_country]['no_NE_attractions_plain'].to_string()
        ne_img = WordCloud().generate(ne)
        no_ne_img = WordCloud().generate(no_ne)
        wordcloud_ne = px.imshow(ne_img)
        wordcloud_no_ne = px.imshow(no_ne_img)
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
