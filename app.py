from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import ast

# ignore setting on copy warning
pd.options.mode.chained_assignment = None  # default='warn'

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
    Output(component_id='attraction_combinations', component_property='figure'),
    #Output(component_id='individual_visualisation', component_property='figure'),
    Input(component_id='global_countries', component_property='value')
    )
def apply_global_filter(global_country):
    # filter Data
    if global_country is not None:
        df_filter = df_prep.loc[df_prep['Country'] == global_country]
        df_filter_pairs = df_comb.loc[df_comb['country'] == global_country]
    else:
        df_filter = df_prep
        df_filter_pairs = df_comb.loc[df_comb['country'] == 'all']

    # build wordclouds
    ne = df_filter['named_entities_spacy_small_plain_unique_dict']
    ne_dict = {}
    for doc in ne:
        doc = ast.literal_eval(doc)
        ne_dict = Counter(ne_dict) + Counter(doc)

    no_ne = df_filter['no_NE_attractions_plain_unique_dict']
    no_ne_dict = {}
    for doc in no_ne:
        doc = ast.literal_eval(doc)
        no_ne_dict = Counter(no_ne_dict) + Counter(doc)

    ne_img = WordCloud().fit_words(ne_dict)
    no_ne_img = WordCloud().fit_words(no_ne_dict)

    wordcloud_ne = px.imshow(ne_img)
    wordcloud_no_ne = px.imshow(no_ne_img)

    # barchart for pairings
    # sort dataframe for support
    df_filter_pairs.sort_values(by='support', ascending=False, inplace=True)
    comb = px.bar(df_filter_pairs.head(10), y='support', x='itemsets', title='popular pairs')
    return wordcloud_ne, wordcloud_no_ne, comb

    # barchart for attraction category count
    n_category = {}


# wordcloud wit NE locations

# wordcloud with attractions (no NE)

# visualisation of top 10 tourist attraction combinations

# own visualisation
# count of attraction types per country
# cornerstones

if __name__ == '__main__':
    app.run_server(debug=True)
