######################## IMPORTS #######################################################################################
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from dash.exceptions import PreventUpdate
import numpy as np
from sklearn.manifold import MDS

from dash.dependencies import Input, Output, State
from plotly import graph_objs as go
from ast import literal_eval
from plotly.graph_objs import *
from datetime import datetime as dt
import requests
import urllib.parse
import ddf_script as dfs


######################## IMPORT AND CLEAN DATA ####################################################################################
TEAM_MEMBERS = (
    "Jennifer", "Henry", "Samuel", "Tim"
)

token = "pk.eyJ1IjoiYXN6ZW5pYSIsImEiOiJjbDRtZHg1aTExMjIzM29ueGZ3aHB6ZXZsIn0.ehBctjUFzibYKM8zjueniw"  # you need your own token

#Topic Modelling
df_lsa = pd.read_csv('output/lsa_example_matrix.csv', header= None)
df_lda = pd.read_csv('output/lda_example_matrix.csv', header= None)
topic_model_list = ["LSA", "LDA"]

#Clustering
df_cluster = pd.read_csv('output/results_2a.csv', index_col=0, converters={'labels':literal_eval})
clustering_model_list = []
index = 1
for ele in df_cluster["cluster_type"]:
    clustering_model_list.append(ele+" "+str(index))
    index +=1

#Data Mapping
df_mapping = pd.read_csv('output/mapping_data_lemma.csv', index_col=0, converters={'no_NE_attractions':literal_eval})

lon = []
lat = []

for place in df_mapping['Place']:
    try:
        place = place.replace(' Travel Guide', '').replace(' and around', '')

        if place == 'Basilique-Cathédrale Marie-Reine-du-Monde':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Montreal') + '?format=json'

        elif place == 'Swedish Lapland':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Kiruna') + '?format=json'

        elif place == 'The Musée du Louvre':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Paris') + '?format=json'

        elif place == 'Champagne and the Ardennes':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Ardennes') + '?format=json'

        elif place == 'The Aeolian Islands':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Aeolian Islands') + '?format=json'

        elif place == 'The Costa Smeralda':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Porto Cervo') + '?format=json'

        elif place == 'The Black Forest':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Titisee-Neustadt') + '?format=json'

        elif place == 'Mount Etna':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Ätna') + '?format=json'

        elif place == 'Provence':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Saint-Tropez') + '?format=json'

        elif place == 'Plaza de Armas':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Cusco') + '?format=json'

        elif place == 'Santa Fe':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Santa Fe USA') + '?format=json'

        elif place == 'The Florida Keys':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Key West') + '?format=json'

        elif place == 'The Capital Region':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Washington') + '?format=json'

        elif place == 'The Big Island':
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Hawaii') + '?format=json'

        elif 'Azur' in place:
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote('Nizza') + '?format=json'

        else:
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(place) + '?format=json'

        response = requests.get(url).json()
        lat.append(response[0]["lat"])
        lon.append(response[0]["lon"])

    except IndexError:
        print('Error')

df_mapping['lat'] = lat
df_mapping['lon'] = lon

print(df_mapping)


######################## APP LAYOUT ####################################################################################
# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},],)
app.title = "Data Science Dashboard - Task 2"

# App layout
app.layout = dbc.Container(
    fluid = True,
    children = [
        html.Header([html.H3("Data Science Dashboard - Task 2")], style={"text-align":"center","padding":"1%"}),

    dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    style={"padding":"1%"},
                    children=[
                        dbc.Row(children=[
                                html.Div(
                                    id="clustering_heading",
                                    children="Clustering Model", style={"text-align":"center"}
                                        ),
                                dbc.Col(
                                    width = 3,
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Div(id="clustering_dropdown_div",
                                                    children=[
                                                        dcc.Dropdown(
                                                            options=clustering_model_list,
                                                            value=clustering_model_list[0],
                                                            id='dropdown_clustering',
                                                            style = {'color': 'black'}
                                                            ),
                                                        dcc.Dropdown(
                                                            options=['Placeholder'],
                                                            id='dropdown_cluster',
                                                            style={'visibility': 'hidden', 'color': 'black'},
                                                            ),

                                                        ]

                                                    )
                                                ]
                                            ),
                                        html.Div(id="clustering_div"),
                                        dbc.Card(dcc.RadioItems(options=['Wordcloud', 'Map'],
                                                       value='Wordcloud',
                                                       id='vis-change'), style={'background-color': '#4e4f50'})

                                    ]
                                ),
                                dbc.Col(
                                    width = 9,
                                    children=[
                                        html.Div(id="clustering_graph_div",
                                            children=[
                                                dcc.Graph(id="clustering_graph", figure=px.scatter(template="plotly_dark"))
                                                ]
                                            )
                                        ]
                                    )


                                ]
                            ),
                        ]
                ),
            ),
            dbc.Col(
                dbc.Card(
                    style={"padding":"1%"},
                    children=[
                        dbc.Row(children=[
                                 html.Div(
                                     id="topic_heading",
                                     children="Topic Model", style={"text-align": "center"}
                                 ),
                                 dbc.Col(
                                     width=3,
                                     id="topic_filters",
                                     children=[
                                         html.Div(
                                             children=[
                                                 html.Div(id="topic_dropdown_div",
                                                          children=[
                                                              dcc.Dropdown(
                                                                  options=topic_model_list,
                                                                  value=topic_model_list[0],
                                                                  id='dropdown_topic',
                                                                  style = {'color': 'black'}
                                                              ),
                                                              dcc.Dropdown(
                                                                  options=clustering_model_list,
                                                                  value=clustering_model_list[0],
                                                                  id='dropdown_topic_cluster',
                                                                  style = {'color': 'black'}
                                                              ),
                                                          ]

                                                          )
                                             ]
                                         ),
                                     ]
                                 ),
                                 dbc.Col(
                                     width=9,
                                     children=[
                                         html.Div(id="topic_graph_div",
                                                  style={"background-color": "black"},
                                                  children=[
                                                      dcc.Graph(id="topic_graph")
                                                  ]
                                                  )
                                     ]
                                 )

                             ]
                        ),
                    ]
                ),
            ),
        ],
    ),
    html.Br(),
    dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    style={"padding":"1%"},
                    children=[
                        dbc.Row(children=[
                                     html.Div(
                                         id="dream_destination_finder_heading",
                                         children="Dream Destination Finder", style={"text-align": "center"}
                                                                              ),
                                     dbc.Col(
                                         width=3,
                                         id="ddf_filters",
                                         children=[
                                             html.Div(
                                                 children=[
                                                     html.Div(id="ddf_input_div",
                                                              children=[
                                                                  dcc.Input(
                                                                      id="input_Jennifer",
                                                                      type="text",
                                                                      placeholder="Preference Jennifer",
                                                                      value="Preference Jennifer"
                                                                  ),
                                                                  dcc.Input(
                                                                      id="input_Henry",
                                                                      type="text",
                                                                      placeholder="Preference Henry",
                                                                      value="Preference Henry"
                                                                  ),
                                                                  dcc.Input(
                                                                      id="input_Samuel",
                                                                      type="text",
                                                                      placeholder="Preference Samuel",
                                                                      value="Preference Samuel"
                                                                  ),
                                                                  dcc.Input(
                                                                      id="input_Tim",
                                                                      type="text",
                                                                      placeholder="Preference Tim",
                                                                      value="Preference Tim"
                                                                  )
                                                              ]

                                                              )
                                                 ]
                                             ),
                                             html.Button('Find the dream destination!', id='go', n_clicks=0),
                                         ]
                                     ),
                                     dbc.Col(
                                         width=9,
                                         children=[
                                             html.Div(id="ddf_graph_div", children=html.Div("Please start the dream destination finder!", style={'text-align':'center', 'padding':'20%'}))
                                         ]
                                     )

                                 ]
                            ),
                        ]
                ),
            ),
            dbc.Col(
                dbc.Card(
                    style={"padding":"1%"},
                    children=[
                        dbc.Row(children=[
                                     html.Div(
                                         id="4v_heading",
                                         children="Dream Destination Mapping", style={"text-align": "center"}
                                     ),
                                     dbc.Col(
                                         width=3,
                                         id="4v_filters",
                                         children=[
                                             html.Div(
                                                 children=[
                                                     html.Div(id="4v_dropdown_div",
                                                              children=[
                                                                  dcc.Checklist(
                                                                      options=[{"label": "label", "value": "value"}],
                                                                      style={'visibility': 'hidden'},
                                                                      id='4v_topic'
                                                                  )
                                                              ]

                                                              )
                                                 ]
                                             ),
                                         ]
                                     ),
                                     dbc.Col(
                                         width=9,
                                         children=[
                                             html.Div(id="ddacm_graph_div",
                                                      children=html.Div("Please start the dream destination finder!", style={'text-align':'center', 'padding':'20%'}),
                                                      )
                                         ]
                                     )

                                 ]
                            ),
                        ]
                    ),
                ),
            ]
        ),
    dbc.Row(html.Div("Authors: Jennifer Hammen, Henry Mangold, Samuel Passauer, Tim Stefany  |    Students at Hochschule der Medien  |    Data from: https://www.roughguides.com",
        style={"font-size":"10px","padding-top":"20px"}
                     ))

])


######################## CALLBACKS #####################################################################################

@app.callback(
    Output(component_id= "topic_graph_div", component_property="children"),
    [Input(component_id="dropdown_topic", component_property="value"),
     Input(component_id="dropdown_topic_cluster", component_property="value")],
)
def update_graph_topic(option_chosen, model):
    """ Updates the topic graph based on the input in the two dropdowns

    """

    if option_chosen == 'LDA':
        mds = MDS(n_components=2, random_state = 2).fit_transform(df_lda)
    elif option_chosen == 'LSA':
        mds = MDS(n_components=2, random_state = 2).fit_transform(df_lsa)

    df_mds = pd.DataFrame(mds, columns=['x','y'])

    name, row = model.split(' ')
    row = int(row)

    df_mds['cluster'] = df_cluster.loc[row, 'labels']
    df_mds['cluster'] = df_mds['cluster'].astype(str)
    df_mds['place'] = df_mapping['Place']

    fig=px.scatter(df_mds, x='x', y='y', color="cluster", hover_data=['place'], template="plotly_dark",
                                       color_discrete_sequence=px.colors.qualitative.Alphabet,)
    fig.update_layout(legend_traceorder="normal")

    return dcc.Graph(figure=fig)


@app.callback(
    [Output(component_id= "dropdown_cluster", component_property="options"),
     Output(component_id= "dropdown_cluster", component_property="value"),
     Output(component_id= "dropdown_cluster", component_property="style")],
    Input(component_id="dropdown_clustering", component_property="value")
)
def create_options(model):
    """ Creates options to choose between different clusters

    """

    name, row = model.split(' ')
    row = int(row)
    clusters = np.unique(df_cluster.loc[row, 'labels'])

    options = []
    for ele in clusters:
        options.append('Cluster '+str(ele))

    print(options)

    return options, options[0], {'visibility': 'visible', 'color': 'black'}


@app.callback(
    [Output(component_id= "clustering_graph", component_property="figure"),
    Output(component_id= "clustering_div", component_property="children")],
    [Input(component_id="dropdown_cluster", component_property="value"),
    Input(component_id="dropdown_clustering", component_property="value"),
    Input(component_id="vis-change", component_property="value")],
)
def create_wordcloud(cluster, model, vis):
    """ Updates the cluster graph and the div based on the input in the two dropdowns

    """

    name, row = model.split(' ')
    row = int(row)
    clusters = df_cluster.loc[row, 'labels']


    if vis == 'Wordcloud':
        name, cluster_nr = cluster.split(' ')
        cluster_nr = int(cluster_nr)

        path = df_cluster.loc[row, 'train_data']
        df_train = pd.read_csv(path, index_col=0)
        df_train['cluster'] = clusters

        df_slice = df_train[df_train['cluster']==cluster_nr]
        places = df_slice['Place']
        df_slice.drop(['Place','cluster'],inplace=True,axis='columns')

        text_dict = {}
        print(df_slice)

        if 'TFIDF' not in path:
            for col in df_slice.columns:
                text_dict[col] = 0
                for row in df_slice[col]:
                    text_dict[col] = row

            text_dict = {x: y for x, y in text_dict.items() if y != 0}

        else:
            for col in df_slice.columns:
                text_dict[col] = np.mean(df_slice[col].to_list())

            text_dict = {x: y for x, y in text_dict.items() if ~np.isnan(y)}

        wordcloud = WordCloud(height=500, width=600, background_color="#0e1012").generate_from_frequencies(text_dict)
        #plt.figure(figsize=(2,1))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        wordcloud_img = px.imshow(wordcloud, template="plotly_dark").update_xaxes(visible=False).update_yaxes(visible=False)
        wordcloud_img.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, hovermode=False)

        list_output = []
        list2_output = []
        for col in df_cluster.columns:
            try:
                is_nan = ~np.isnan(df_cluster.loc[row, col])
            except:
                is_nan = True
            if col not in ['train_data', 'labels'] and is_nan:
                list2_output.append(html.Div(col+': '+str(df_cluster.loc[row, col])),)
        list_output.append(dbc.Card(list2_output, style={'background-color': '#565656'}),)
        list_output.append(dbc.Card('Places:    '+', '.join(places),style={'background-color': '#474848'}),)

        return wordcloud_img, list_output

    elif vis == 'Map':
        df_mapping['clusters'] = clusters
        df_mapping['clusters'] = df_mapping['clusters'].astype('str')

        place_categories = np.unique(df_mapping['clusters'])
        cluster_color_dict = dict()

        for ele in range(len(place_categories)):
            cluster_color_dict[place_categories[ele]] = px.colors.qualitative.Alphabet[ele]

        df_mapping['color'] = df_mapping['clusters'].replace(to_replace=cluster_color_dict)

        fig = go.Figure(go.Scattermapbox(
            mode="markers+text",
            lon=df_mapping['lon'], lat=df_mapping['lat'],
            customdata=np.stack((df_mapping['Place'], df_mapping['clusters']), axis=-1),
            hovertemplate='<b>Place: %{customdata[0]}</b><br>Cluster: %{customdata[1]}</br><extra></extra>',
            marker=dict(#color=df_mapping['color'],
                    opacity=0.5,size= 10),
            text=df_mapping['Place'], textposition="bottom right"),

        )

        fig.update_layout(
            mapbox={
                'accesstoken': token,
                'style': "dark",
                # 'zoom': 0.7
            },
            #showlegend=True,
            template="plotly_dark")

        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

        list_output = []
        list2_output = []
        for col in df_cluster.columns:
            try:
                is_nan = ~np.isnan(df_cluster.loc[row, col])
            except:
                is_nan = True
            if col not in ['train_data', 'labels'] and is_nan:
                list2_output.append(html.Div(col+': '+str(df_cluster.loc[row, col])),)
        list_output.append(dbc.Card(list2_output, style={'background-color': '#565656'}),)

        return fig, list_output


@app.callback(
    [Output(component_id="ddf_graph_div", component_property="children"),
     Output(component_id= "4v_topic", component_property="options"),
     Output(component_id= "4v_topic", component_property="value"),
     Output(component_id= "4v_topic", component_property="style")],
     Input(component_id= "go", component_property="n_clicks"),
    [State(component_id="input_Jennifer", component_property="value"),
     State(component_id="input_Henry", component_property="value"),
     State(component_id="input_Samuel", component_property="value"),
     State(component_id="input_Tim", component_property="value")]
)
def update_graph(clicks, preference_Jennifer, preference_Henry, preference_Samuel, preference_Tim):
    """ Creates from the input preferences the dream destinations and updates the output graph and div

    """

    if clicks != 0:
        preferences = []
        preferences.append(preference_Jennifer)
        preferences.append(preference_Henry)
        preferences.append(preference_Samuel)
        preferences.append(preference_Tim)

        df_destinations = dfs.find_dream_destination()

        top3_destinations = df_destinations.iloc[-3:,:]

        df_slice = df_mapping[df_mapping['Place'].isin(top3_destinations['place'])]
        df_slice['score'] = top3_destinations['total_score']

        print(df_slice)

        fig = go.Figure(go.Scattermapbox(
            mode="markers+text",
            lon=df_slice['lon'], lat=df_slice['lat'],
            marker={'size': 10, 'symbol': "airport"},
            text=df_slice['Place']+df_slice['score'], textposition="bottom right"))

        fig.update_layout(
            mapbox={
                'accesstoken': token,
                'style': "dark",
                # 'zoom': 0.7
            },
            showlegend=False,
            template="plotly_dark")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

        options = []
        values = []

        for child in top3_destinations:
            options.append({'label': child, 'value': child})
            values.append(child)

        return dcc.Graph(figure=fig), options, values, {'visibility': 'visible','color': 'black'}

    else:
        raise PreventUpdate


@app.callback(
    Output(component_id= "ddacm_graph_div", component_property="children"),
    Input(component_id="4v_topic", component_property="value")

)
def update_graph_mapper(values):
    """ Updates the mapping graph based on the dream destination

    """

    if values != None:
        print(values)

        df_slice = df_mapping[df_mapping['Place'].isin(values)][['Place', 'no_NE_attractions']]

        df = pd.DataFrame()
        indx = 0
        for index, row in df_slice.iterrows():
            dictio = {}
            setio = set(row['no_NE_attractions'])
            print(setio)
            for ele in setio:

                if ele[1] in dictio.keys():
                    dictio[ele[1]] = dictio[ele[1]] + 1
                else:
                    dictio[ele[1]] = 1

            for key in dictio:
                df.loc[indx, 'attraction'] = key
                df.loc[indx, 'count'] = dictio[key]
                df.loc[indx, 'place'] = row['Place']
                indx += 1

        fig = px.histogram(df, x="attraction", y="count",
                     color='place', barmode='group',template="plotly_dark")
        return dcc.Graph(figure=fig)

    else:
        raise PreventUpdate


######################## RUN APP #######################################################################################

if __name__ == "__main__":
    app.run_server(debug=True)