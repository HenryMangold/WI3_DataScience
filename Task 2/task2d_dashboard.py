# import libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
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
                                                dcc.Dropdown(
                                                    options=[{"label": "label", "value": "value"}],
                                                    placeholder='Select clustering algorithm',
                                                    id='dropdown_clustering'
                                                )
                                            ]

                                    )
                                ]
                            ),
                            html.Div(
                            "Div 1.2",
                            id="Div 1.2")
                        ]
                    ),
                    html.Div(
                        style={"display": "inline-block", "width":"70%"},
                        children=[
                        html.Div(id="clustering_graph_div",
                                 children=[
                                     dcc.Graph(id="clustering_graph", figure=px.scatter())
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
                                                      options=[{"label": "label", "value": "value"}],
                                                      placeholder='Select topic modeling algorithm',
                                                      id='dropdown_topic'
                                                  )
                                              ]

                                              )
                                 ]
                             ),
                             html.Div(
                                 "Selected topic model",
                                 id="topic_model_selected")
                         ]
                     ),
                     html.Div(
                         style={"display": "inline-block", "width": "70%"},
                         children=[
                             html.Div(id="topic_graph_div",
                                      children=[
                                          dcc.Graph(id="topic_graph", figure={})
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
                                                          id="input_{}".format(_),
                                                          type="text",
                                                          placeholder="Preference of {}".format(_),
                                                      )
                                                      for _ in TEAM_MEMBERS
                                                  ]

                                                  )
                                     ]
                                 ),
                                 html.Div(
                                     "Dream Destination Finder",
                                     id="ddf")
                             ]
                         ),
                         html.Div(
                             style={"display": "inline-block", "width": "70%"},
                             children=[
                                 html.Div(id="ddf_graph_div",
                                          children=[
                                              dcc.Graph(id="ddf_graph", figure={})
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
                                              dcc.Graph(id="ddacm_graph", figure=px.bar())
                                          ]
                                          )
                             ]
                         )

                     ]
                     ),
        ]
    ),
    dbc.Row(
        "Autoren: Jennifer Hammen, Henry Mangold, Samuel Passauer, Tim Stefany",
        style={"text-align":"center","height":"2%"}
    )

])

# Callbacks
'''@app.callback(
    [Output(component_id="clustering_container"), Output(component_id="clustering_graph"),
     Input(component_id="clustering_dropdown")]
)

@app.callback(
    [Output(component_id="topic_container"), Output(component_id="topic_graph"),
     Input(component_id="topic_dropdown")]
)

@app.callback(
    [Output(component_id="ddf_container"), Output(component_id="ddf_graph"),
     Input(component_id="ddf_dropdown")]
)'''



def update_graph():
    graph= 'graph'
    print(graph)

if __name__ == "__main__":
    app.run_server(debug=True)