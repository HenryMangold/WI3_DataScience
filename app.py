from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import ast
import re
import numpy as np

# ignore setting on copy warning
pd.options.mode.chained_assignment = None  # default='warn'

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
template = "simple_white"

# read csv files to dataframes
df_prep = pd.read_csv('data/results_df_prep.csv')
df_comb = pd.read_csv('data/results_top_combinations.csv')

# create list with countries for dropdown filter
countries = df_prep['Country'].unique().tolist()
prep_countries = []
for country in countries:
    if country == 'usa':
        prep_countries.append(country.upper())
    else:
        prep_countries.append(country.capitalize())

# create HTML layout including all input and output components
app.layout = dbc.Container([
    html.Div([
        dbc.Row(dbc.Col(
            html.Div([
                html.H1('Dashboard Data Science - Travel Location Finder', style={'padding': '1em', 'text-align': 'center'}),
                # country filter for entire dashboard
                html.Div([
                    html.H5('Filter'),
                    dcc.Dropdown(
                        prep_countries,
                        searchable=True,
                        id='global_countries',
                        style={'width': '50%'},
                        placeholder='Select Country ...'
                    )
                ])
            ])
        )),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        html.H4('Named Entities', style={'padding-top': '1.5em'}),
                        dbc.Checklist(
                            options=[
                                {"label": "Average over documents", "value": 1}
                            ],
                            value=False,
                            id="absolute_ne",
                            switch=True,
                            style={'fontSize': 12}
                        ),
                        dcc.Graph(
                            id='wordcloud_ne',
                            config={"displayModeBar": False}
                        )
                    ])
                ),
                dbc.Col(
                    html.Div([
                        html.H4('No Named Entities', style={'padding-top': '1.5em'}),
                        dbc.Checklist(
                            options=[
                                {"label": "Average over documents", "value": 1}
                            ],
                            value=False,
                            id="absolute_no_ne",
                            switch=True,
                            style={'fontSize': 12}
                        ),
                        dcc.Graph(
                            id='wordcloud_no_ne',
                            config={"displayModeBar": False}
                        )
                    ])
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        html.H4('Popular Combinations'),
                        dcc.Graph(
                            id='attraction_combinations',
                            config={"displayModeBar": False}
                        )
                    ])
                ),
                dbc.Col(
                    html.Div([
                        html.H4('Popular Categories'),
                        dcc.Graph(
                            id='popular_categories',
                            config={"displayModeBar": False}
                        )
                    ])
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        html.H4('Cornerstones'),
                        dcc.Graph(
                            id='cornerstones',
                            config={"displayModeBar": False}
                        )
                    ])
                )
            ]
        ),
    ])
])


# callback for global continent filtering
@app.callback(
    [
        Output(component_id='wordcloud_ne', component_property='figure'),
        Output(component_id='wordcloud_no_ne', component_property='figure'),
        Output(component_id='attraction_combinations', component_property='figure'),
        Output(component_id='cornerstones', component_property='figure'),
        Output(component_id='popular_categories', component_property='figure')
    ],
    [
        Input(component_id='global_countries', component_property='value'),
        Input(component_id='absolute_ne', component_property='value'),
        Input(component_id='absolute_no_ne', component_property='value')
    ]

)
def apply_global_filter(global_country, absolute_ne, absolute_no_ne):
    # filter Data
    if global_country is not None:
        global_country = global_country.lower()
        df_filter = df_prep.loc[df_prep['Country'] == global_country]
        df_filter_pairs = df_comb.loc[df_comb['country'] == global_country]
    else:
        df_filter = df_prep
        df_filter_pairs = df_comb.loc[df_comb['country'] == 'all']
    total_rows = len(df_filter.index)

    # build wordclouds
    ne_dict = {}
    for doc in df_filter['named_entities_spacy_small_plain_unique_dict']:
        doc = ast.literal_eval(doc)
        ne_dict = Counter(ne_dict) + Counter(doc)
    # watch for toggle switch
    if absolute_ne:
        ne_dict = {k: v / total_rows for k, v in ne_dict.items()}

    no_ne_dict = {}
    for doc in df_filter['no_NE_attractions_plain_unique_dict']:
        doc = ast.literal_eval(doc)
        no_ne_dict = Counter(no_ne_dict) + Counter(doc)
    # watch for toggle switch
    if absolute_no_ne:
        no_ne_dict = {k: v / total_rows for k, v in no_ne_dict.items()}

    # generate wordcloud from dict
    ne_img = WordCloud(background_color='white', width=1080, height=540, random_state=100).fit_words(ne_dict)
    no_ne_img = WordCloud(background_color='white', width=1080, height=540, random_state=100).fit_words(no_ne_dict)

    # generate figure for images
    wordcloud_ne = px.imshow(ne_img, template=template) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    wordcloud_ne.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, hovermode=False)
    wordcloud_no_ne = px.imshow(no_ne_img, template=template) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    wordcloud_no_ne.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, hovermode=False)

    # barchart for pairings
    # sort dataframe for support and remove ' in tuple
    df_filter_pairs.sort_values(by='support', ascending=False, inplace=True)
    df_filter_pairs['itemsets'] = df_filter_pairs['itemsets'].str.replace("'", '')
    # create horizontal barchart with ascending order
    comb = px.bar(df_filter_pairs.head(10), y='itemsets', x='support', template=template, orientation='h')
    comb.update_layout(yaxis_title='Itemset', xaxis_title='Support-Value', yaxis=dict(titlefont=dict(size=12)),
                       xaxis=dict(titlefont=dict(size=12)))
    comb.update_layout(yaxis={'categoryorder': 'total ascending'})

    # build cornerstones
    # build list with all named entities from keys in wordcloud dict
    ne_unique = ne_dict.keys()
    # find number of rows where Substring is found and build dataframe
    cs = {}
    for search in ne_unique:
        cs[search] = df_filter["named_entities_spacy_small_plain_unique"].str.contains(re.escape(search)).sum()
    cs = pd.DataFrame.from_dict(cs, orient='index')
    cs.sort_values(by=0, ascending=False, inplace=True)

    # create barchart for cornerstones
    corner_stones = px.bar(cs.head(10), y=cs.head(10).index, x=0, template=template, orientation='h')
    corner_stones.update_layout(xaxis_title='Number of Documents', yaxis_title='Named Entity',
                                yaxis=dict(titlefont=dict(size=12)), xaxis=dict(titlefont=dict(size=12)))
    corner_stones.update_layout(yaxis={'categoryorder': 'total ascending'})

    # barchart for attraction category count
    n_category = {}
    for doc in df_filter['no_NE_attractions_categories_count']:
        doc = ast.literal_eval(doc)
        n_category = Counter(n_category) + Counter(doc)
    cat = pd.DataFrame.from_dict(n_category, orient='index')
    cat.sort_values(by=0, ascending=False, inplace=True)
    # create barchart
    cat_count = px.bar(cat.head(10), y=cat.head(10).index, x=0, template=template, orientation='h')
    cat_count.update_layout(xaxis_title='Number of Occurrences', yaxis_title='Category',
                            yaxis=dict(titlefont=dict(size=12)), xaxis=dict(titlefont=dict(size=12)))
    cat_count.update_layout(yaxis={'categoryorder': 'total ascending'})

    return wordcloud_ne, wordcloud_no_ne, comb, corner_stones, cat_count


if __name__ == '__main__':
    app.run_server(debug=True)
