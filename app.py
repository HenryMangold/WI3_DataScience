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
                html.H1('Dashboard Data Science - Travel Location Finder',
                        style={'padding': '1em', 'text-align': 'center'}),
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
                        dbc.RadioItems(
                            options=[
                                {"label": "Absolute number", "value": False},
                                {"label": "Number of documents", "value": True}
                            ],
                            value=False,
                            id="n_doc_ne",
                            inline=True,
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
                        dbc.RadioItems(
                            options=[
                                {"label": "Absolute number", "value": False},
                                {"label": "Number of documents", "value": True}
                            ],
                            value=False,
                            id="n_doc_no_ne",
                            inline=True,
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
        dcc.Store(id='df_filter', storage_type='memory'),
        dcc.Store(id='df_filter_pairs', storage_type='memory'),
    ])
])

# callback for updating data filters
@app.callback(
    [
        Output('df_filter', 'data'),
        Output('df_filter_pairs', 'data')
    ],
    [
        Input('global_countries', 'value')
    ]
)
def global_filter(country):
    # filter Data
    if country is not None:
        country = country.lower()
        df_filter = df_prep.loc[df_prep['Country'] == country]
        df_filter_pairs = df_comb.loc[df_comb['country'] == country]
    else:
        df_filter = df_prep
        df_filter_pairs = df_comb.loc[df_comb['country'] == 'all']

    # convert dataframes to dict for storing
    df_filter = df_filter.to_dict(orient='dict')
    df_filter_pairs = df_filter_pairs.to_dict(orient='dict')

    return df_filter, df_filter_pairs


# wordclouds
# change dict to number of documents with 'ne' and 'no ne'
def change_to_n_doc(dictionary, search_column):
    for search in dictionary.keys():
        dictionary[search] = search_column.str.contains(re.escape(search)).sum()
    return dictionary


# merge all dicts in dataframe for 'ne' and 'no ne'
def merge_dicts(column):
    dictionary = {}
    for doc in column:
        doc = ast.literal_eval(doc)
        dictionary = Counter(dictionary) + Counter(doc)
    return dictionary


# generate wordcloud from dicts
def generate_wordcloud_fig(dictionary):
    img = WordCloud(background_color='white', width=1080, height=540, random_state=100).fit_words(dictionary)

    # generate figure for images
    wordcloud = px.imshow(img, template=template) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    wordcloud.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, hovermode=False)

    return wordcloud


@app.callback(
    Output('wordcloud_ne', 'figure'),
    [
        Input('df_filter', 'data'),
        Input('n_doc_ne', 'value')
    ]
)
def update_wordcloud_ne(df_filter, n_doc_ne):
    # read dataframes from dict
    df_filter = pd.DataFrame.from_dict(df_filter)

    ne_dict = merge_dicts(df_filter['named_entities_spacy_small_plain_unique_dict'])

    # change depending on radio button
    if n_doc_ne:
        ne_dict = change_to_n_doc(ne_dict, df_filter['named_entities_spacy_small_plain_unique'])

    # generate figure for wordcloud
    wordcloud_ne = generate_wordcloud_fig(ne_dict)

    return wordcloud_ne


@app.callback(
    Output('wordcloud_no_ne', 'figure'),
    [
        Input('df_filter', 'data'),
        Input('n_doc_no_ne', 'value')
    ]
)
def update_wordcloud_no_ne(df_filter, n_doc_no_ne):
    # read dataframes from dict
    df_filter = pd.DataFrame.from_dict(df_filter)

    no_ne_dict = merge_dicts(df_filter['no_NE_attractions_plain_unique_dict'])

    # change depending on radio button
    if n_doc_no_ne:
        no_ne_dict = change_to_n_doc(no_ne_dict, df_filter['no_NE_attractions_plain_unique'])

    # generate wordlcloud figure
    wordcloud_no_ne = generate_wordcloud_fig(no_ne_dict)

    return wordcloud_no_ne


# callback for generating bar charts
@app.callback(
    [
        Output(component_id='attraction_combinations', component_property='figure'),
        Output(component_id='popular_categories', component_property='figure')
    ],
    [
        Input(component_id='df_filter', component_property='data'),
        Input(component_id='df_filter_pairs', component_property='data')
    ]

)
def generate_bar_charts(df_filter, df_filter_pairs):
    # create dataframes form dicts
    df_filter = pd.DataFrame.from_dict(df_filter)
    df_filter_pairs = pd.DataFrame.from_dict(df_filter_pairs)

    # barchart for pairings
    # sort dataframe for support and remove ' in tuple
    df_filter_pairs.sort_values(by='support', ascending=False, inplace=True)
    df_filter_pairs['itemsets'] = df_filter_pairs['itemsets'].str.replace("'", '')
    # create horizontal barchart with ascending order
    comb = px.bar(df_filter_pairs.head(10), y='itemsets', x='support', template=template, orientation='h')
    comb.update_layout(yaxis_title='Itemset', xaxis_title='Support-Value', yaxis=dict(titlefont=dict(size=12)),
                       xaxis=dict(titlefont=dict(size=12)))
    comb.update_layout(yaxis={'categoryorder': 'total ascending'})

    # barchart for attraction category count
    # merge dicts for category count
    n_category = {}
    for doc in df_filter['no_NE_attractions_categories_count']:
        doc = ast.literal_eval(doc)
        n_category = Counter(n_category) + Counter(doc)
    cat = pd.DataFrame.from_dict(n_category, orient='index')
    cat.sort_values(by=0, ascending=False, inplace=True)
    cat_label_df = pd.DataFrame(cat.head(10).index, columns={'categories'})
    labels = cat_label_df['categories'].values
    cat_labels = [s.replace('_', ' ') for s in labels]
    cat_labels = np.char.capitalize(cat_labels)
    # create barchart
    cat_count = px.bar(cat.head(10), y=cat_labels, x=0, template=template, orientation='h')
    cat_count.update_layout(xaxis_title='Number of Occurrences', yaxis_title='Category',
                            yaxis=dict(titlefont=dict(size=12)), xaxis=dict(titlefont=dict(size=12)))
    cat_count.update_layout(yaxis={'categoryorder': 'total ascending'})

    return comb, cat_count


if __name__ == '__main__':
    app.run_server(debug=True)
