# import libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np

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
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.CYBORG])
app.title = "Data Science Dashboard - Task 2"

# App layout
app.layout = dbc.Container([

    dbc.Row(
        html.H1("Data Science Dashboard - Task 2", style={"text-align":"center","height":"8%"})
    ),
    dbc.Row(
        style={"text-align":"center","height":"40%"},
        children=[
            html.Div(id="clustering_model",
                style={"display": "inline-block", "width":"50%"},
                children=[
                    html.Div(
                        id="clustering_heading",
                        children=[
                            html.H2("Clustering Model", style={"text-align":"center"})
                        ]
                    ),
                    html.Div(
                        style={"display": "inline-block", "width":"30%"},
                        id="clustering_filters",
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
                            html.Div(
                            id="clustering_div", children=[])
                        ]
                    ),
                    html.Div(
                        style={"display": "inline-block", "width":"70%"},
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
            html.Div(id="topic_model",
                 style={"display": "inline-block", "width": "50%"},
                 children=[
                     html.Div(
                         id="topic_heading",
                         children=[
                             html.H2("Topic Model", style={"text-align": "center"})
                         ]
                     ),
                     html.Div(
                         style={"display": "inline-block", "width": "30%"},
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
                     html.Div(
                         style={"display": "inline-block", "width": "70%", "background-color": "black"},
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
    dbc.Row(
        style={"text-align":"center","height":"40%"},
        children=[
            html.Div(id="dream_destination_finder_div",
                     style={"display": "inline-block", "width": "50%"},
                     children=[
                         html.Div(
                             id="dream_destination_finder_heading",
                             children=[
                                 html.H2("Dream Destination Finder", style={"text-align": "center"})
                             ]
                         ),
                         html.Div(
                             style={"display": "inline-block", "width": "30%"},
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
                         html.Div(
                             style={"display": "inline-block", "width": "70%"},
                             children=[
                                 html.Div(id="ddf_graph_div",
                                          children=[
                                              dcc.Graph(id="ddf_graph", figure=px.bar(df, x="day", y="total_bill", color="sex", template="plotly_dark"))
                                          ]
                                          )
                             ]
                         )

                     ]
                     ),
            html.Div(id="fourth_visualization",
                     style={"display": "inline-block", "width": "50%"},
                     children=[
                         html.Div(
                             id="4v_heading",
                             children=[
                                 html.H2("Dream Destination Mapping", style={"text-align": "center"})
                             ]
                         ),
                         html.Div(
                             style={"display": "inline-block", "width": "30%"},
                             id="4v_filters",
                             children=[
                                 html.Div(
                                     children=[
                                         html.Div(id="4v_dropdown_div",
                                                  children=[
                                                      dcc.Dropdown(
                                                          options=[{"label": "label", "value": "value"}],
                                                          placeholder='XXX',
                                                          id='4v_topic'
                                                      )
                                                  ]

                                                  )
                                     ]
                                 ),
                                 html.Div(
                                     "Dream Destination Attraction Category Mapping",
                                     id="ddacm")
                             ]
                         ),
                         html.Div(
                             style={"display": "inline-block", "width": "70%"},
                             children=[
                                 html.Div(id="ddacm_graph_div",
                                          children=[
                                              dcc.Graph(
                                                  id="ddacm_graph",
                                                  figure=px.bar(df, x="day", y="total_bill", color="sex", template="plotly_dark"))
                                          ]
                                          )
                             ]
                         )

                     ]
                     ),
        ]
    ),
    dbc.Row(
        "Authors: Jennifer Hammen, Henry Mangold, Samuel Passauer, Tim Stefany  |    Students at Hochschule der Medien  |    Data from: https://www.roughguides.com",
        style={"text-align":"center","height":"2%"}
    )

])

"""         Dashboard           """
# Callback Clustering Model
@app.callback(
    Output(component_id= "clustering_div", component_property="children"),
    Input(component_id="dropdown_clustering", component_property="value")

)
def update_graph(option_chosen):
    output=option_chosen
    return output

# Callback Topic Model - wordcloud
@app.callback(
    [Output(component_id= "topic_graph", component_property="figure"),
    Output(component_id= "topic_model_selected", component_property="children")],
    Input(component_id="dropdown_topic", component_property="value")
)

def create_wordcloud(topic_model):
    input_wordcloud = str(topic_model) + "rwveve wrfwrg fgwrgwrg gwrg wgw rwgbwrbwrbrb"
    wordcloud = WordCloud(height=500, width=500).generate(input_wordcloud)
    #plt.figure(figsize=(2,1))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    wordcloud_img = px.imshow(wordcloud).update_xaxes(visible=False).update_yaxes(visible=False)
    wordcloud_img.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, hovermode=False)
    return wordcloud_img, str(topic_model)


# Callback Dream  Destination Finder
@app.callback(
    Output(component_id= "ddf_container", component_property="children"),
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
    return preferences






if __name__ == "__main__":
    app.run_server(debug=True)