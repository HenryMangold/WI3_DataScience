# import libraries
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

from dash.dependencies import Input, Output
from plotly import graph_objs as go
from plotly.graph_objs import *
from datetime import datetime as dt


# import and clean data

TEAM_MEMBERS = (
    "Jennifer", "Henry", "Samuel", "Tim"
)

clustering_model_list  = ["model 1", "model 2", "model 3", "model 4"]
topic_model_list = ["model 1", "model 2", "model 3", "model 4"]


df = px.data.tips()


"""         Dashboard           """
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
        #style={"text-align":"center","height":"40%"},
                        dbc.Row(#id="clustering_model",
                            children=[
                                html.Div(
                                    id="clustering_heading",
                                    children="Clustering Model", style={"text-align":"center"}
                                        ),
                                dbc.Col(
                                    width = 3,
                                    #id="clustering_filters",
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Div(id="clustering_dropdown_div",
                                                    children=[
                                                        dcc.RadioItems(
                                                            options=clustering_model_list,
                                                            #placeholder='Select clustering algorithm',
                                                            id='dropdown_clustering'
                                                            )
                                                        ]

                                                    )
                                                ]
                                            ),
                                        html.Div(id="clustering_div")
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
                        dbc.Row(#id="topic_model",
                             children=[
                                 html.Div(
                                     id="topic_heading",
                                     children="Topic Model", style={"text-align": "center"}
                                 ),
                                 dbc.Col(
                                     width=3,
                                     #style={"display": "inline-block", "width": "30%"},
                                     id="topic_filters",
                                     children=[
                                         html.Div(
                                             children=[
                                                 html.Div(id="topic_dropdown_div",
                                                          children=[
                                                              dcc.Dropdown(
                                                                  options=topic_model_list,
                                                                  placeholder='Select topic modeling algorithm',
                                                                  id='dropdown_topic'
                                                              )
                                                          ]

                                                          )
                                             ]
                                         ),
                                         html.Div(
                                             id="topic_model_selected", children=[]
                                             )
                                     ]
                                 ),
                                 dbc.Col(
                                     width=9,
                                     #style={"display": "inline-block", "width": "70%", "background-color": "black"},
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
        #style={"text-align":"center","height":"40%"},
                        dbc.Row(#id="dream_destination_finder_div",
                                 children=[
                                     html.Div(
                                         id="dream_destination_finder_heading",
                                         children="Dream Destination Finder", style={"text-align": "center"}
                                                                              ),
                                     dbc.Col(
                                         width=3,
                                         #style={"display": "inline-block", "width": "30%"},
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
                                             html.Div(
                                                 id="ddf_container", children=[])
                                         ]
                                     ),
                                     dbc.Col(
                                         width=9,
                                         #style={"display": "inline-block", "width": "70%"},
                                         children=[
                                             html.Div(id="ddf_graph_div", children="Please start the dream destination finder!")
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
                        dbc.Row(#id="fourth_visualization",
                                 children=[
                                     html.Div(
                                         id="4v_heading",
                                         children="Dream Destination Mapping", style={"text-align": "center"}
                                     ),
                                     dbc.Col(
                                         width=3,
                                         #style={"display": "inline-block", "width": "30%"},
                                         id="4v_filters",
                                         children=[
                                             html.Div(
                                                 children=[
                                                     html.Div(id="4v_dropdown_div",
                                                              children=[
                                                                  dcc.Checklist(
                                                                      options=[{"label": "label", "value": "value"}],
                                                                      #placeholder='XXX',
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
                                         #style={"display": "inline-block", "width": "70%"},
                                         children=[
                                             html.Div(id="ddacm_graph_div",
                                                      children="Please start the dream destination finder!", style={'text-align':'center'}
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

"""         Dashboard           """
# Callback Clustering Model
@app.callback(
    [Output(component_id= "topic_model_selected", component_property="children"),
     Output(component_id= "topic_graph_div", component_property="children")],
    Input(component_id="dropdown_topic", component_property="value")
)
def update_graph_topic(option_chosen):
    output=option_chosen
    return output, dcc.Graph(figure=px.scatter(template="plotly_dark"))


# Callback Topic Model - wordcloud
@app.callback(
    [Output(component_id= "clustering_graph", component_property="figure"),
    Output(component_id= "clustering_div", component_property="children")],
    Input(component_id="dropdown_clustering", component_property="value")
)
def create_wordcloud(topic_model):
    input_wordcloud = "bear cat dog animal bird fish shark seehorse monkey dolphin elephant fox"
    wordcloud = WordCloud(height=500, width=600, background_color="#0e1012").generate(input_wordcloud)
    #plt.figure(figsize=(2,1))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    wordcloud_img = px.imshow(wordcloud, template="plotly_dark").update_xaxes(visible=False).update_yaxes(visible=False)
    wordcloud_img.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, hovermode=False)
    return wordcloud_img, str(topic_model)


# Callback Dream  Destination Finder
@app.callback(
    [Output(component_id= "ddf_container", component_property="children"),
     Output(component_id="ddf_graph_div", component_property="children")],
    [Input(component_id="input_Jennifer", component_property="value"),
     Input(component_id="input_Henry", component_property="value"),
     Input(component_id="input_Samuel", component_property="value"),
     Input(component_id="input_Tim", component_property="value")]
)
def update_graph(preference_Jennifer, preference_Henry, preference_Samuel, preference_Tim):
    preferences = []
    preferences.append(preference_Jennifer)
    preferences.append(preference_Henry)
    preferences.append(preference_Samuel)
    preferences.append(preference_Tim)
    print(type(preferences))
    print(preferences)
    token = "pk.eyJ1IjoiYXN6ZW5pYSIsImEiOiJjbDRtZHg1aTExMjIzM29ueGZ3aHB6ZXZsIn0.ehBctjUFzibYKM8zjueniw"  # you need your own token

    fig = go.Figure(go.Scattermapbox(
        mode="markers+text",
        lon=[-86.84656, 13.404954], lat=[21.17429, 52.520008],
        marker={'size': 10, 'symbol': ["airport", "airport"]},
        text=['Cancun', 'Berlin'], textposition="bottom right"))

    fig.update_layout(
        mapbox={
            'accesstoken': token,
            'style': "dark",
            #'zoom': 0.7
    },
        showlegend=False,
        template="plotly_dark")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    #px.set_mapbox_access_token()
    #df = pd.DataFrame({'centroid_lat': [21.17429, 52.520008], 'centroid_lon': [-86.84656, 13.404954], 'place': ['Cancun', 'Berlin']})
    #fig = px.scatter_mapbox(df, lat="centroid_lat", lon="centroid_lon", hover_name="place", template="plotly_dark")
                            #color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
    return preferences, dcc.Graph(figure=fig)


# Callback Dream Destination Filters
@app.callback(
    Output(component_id= "4v_topic", component_property="options"),
    Input(component_id="ddf_container", component_property="children"),


)
def update_options(children):

    options = []

    for child in children:
        options.append({'label' : child, 'value' : child})
    return options


# Callback Dream  Destination Mapper
@app.callback(
    Output(component_id= "ddacm_graph_div", component_property="children"),
    Input(component_id="4v_topic", component_property="value")

)
def update_graph_mapper(values):
    df = pd.DataFrame(
        {'attraction': ['Food', 'Animal', 'Landscape', 'Food', 'Animal', 'Landscape'], 'count': [1,2,3,4,5,6], 'place': ['Cancun','Cancun','Cancun', 'Berlin', 'Berlin', 'Berlin']})
    fig = px.histogram(df, x="attraction", y="count",
                 color='place', barmode='group',template="plotly_dark")
    return dcc.Graph(figure=fig)





if __name__ == "__main__":
    app.run_server(debug=True)