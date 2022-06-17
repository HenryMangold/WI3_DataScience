# import libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import stylecloud as sc
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import PIL.Image
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
    Output(component_id= "clustering_container", component_property="children"),
    Input(component_id="dropdown_clustering", component_property="value")

)
def update_graph(input):
    output=input
    return output

# Callback Topic Model
@app.callback(
    Output(component_id= "topic_graph", component_property="figure"),
    [Input(component_id="input_Jennifer", component_property="value"),
     Input(component_id="input_Henry", component_property="value"),
     Input(component_id="input_Samuel", component_property="value"),
     Input(component_id="input_Tim", component_property="value")]
)
def update_graph(preference_Jennifer, preference_Henry, preference_Samuel, preference_Tim):
    input_wordcloud = str(preference_Jennifer) + str(preference_Henry) + str(preference_Samuel) + str(preference_Tim)
    print(input_wordcloud)
    wordcloud = WordCloud().generate(input_wordcloud)
    plt.figure(figsize=(2,1))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    wordcloud_img = px.imshow(wordcloud).update_xaxes(visible=False).update_yaxes(visible=False)
    wordcloud_img.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, hovermode=False)
    return wordcloud_img

#Code to create wordclouds
"""sample_text = "Though its cities draw the most tourists – New York, New Orleans, Miami, Los Angeles and San Francisco are all incredible destinations in their own right – America is above all a land of stunningly diverse and achingly beautiful landscapes . In one nation you have the mighty Rockies and spectacular Cascades, the vast, mythic desert landscapes of the Southwest, the endless, rolling plains of Texas and Kansas, the tropical beaches and Everglades of Florida, the giant redwoods of California and the sleepy, pristine villages of New England. You can soak up the mesmerizing vistas in Crater Lake, Yellowstone and Yosemite national parks, stand in awe at the Grand Canyon, hike the Black Hills, cruise the Great Lakes, paddle in the Mississippi, surf the gnarly breaks of Oahu and get lost in the vast wilderness of Alaska. Or you could easily plan a trip that focuses on the out-of-the-way hamlets, remote prairies, eerie ghost towns and forgotten byways that are every bit as “American” as its showpiece icons and monuments. The sheer size of the country prevents any sort of overarching statement about the typical American experience, just as the diversity of its people undercuts any notion of the typical American. Icons as diverse as Mohammed Ali, Louis Armstrong, Sitting Bull, Hillary Clinton, Michael Jordan, Madonna, Martin Luther King, Abraham Lincoln, Elvis Presley, Mark Twain, John Wayne and Walt Disney continue to inspire and entertain the world, and everyone has heard of the blues, country and western, jazz, rock ’n’ roll and hip-hop – all American musical innovations. There are Irish Americans, Italian Americans, African Americans, Chinese Americans and Latinos, Texan cowboys and Bronx hustlers, Seattle hipsters and Alabama pastors, New England fishermen, Las Vegas showgirls and Hawaiian surfers. Though it often sounds clichéd to foreigners, the only thing that holds this bizarre federation together is the oft-maligned “American Dream”. While the USA is one of the world’s oldest still-functioning democracies and the roots of its European presence go back to the 1500s, the palpable sense of newness here creates an odd sort of optimism, wherein anything seems possible and fortune can strike at any moment.Indeed, aspects of American culture can be difficult for many visitors to understand, despite the apparent familiarity: its obsession with guns; the widely held belief that “government” is bad; the real, genuine pride in the American Revolution and the US Constitution, two hundred years on; the equally genuine belief that the USA is the “greatest country on earth”; the wild grandstanding of its politicians (especially at election time); and the bewildering contradiction of its great liberal and open-minded traditions with laissez-faire capitalism and extreme cultural and religious conservatism. That’s America: diverse, challenging, beguiling, maddening at times, but always entertaining and always changing. And while there is no such thing as a typical American person or landscape, there can be few places where strangers can feel so confident of a warm reception. "
mask = np.array(PIL.Image.open("Weltkarte.jpeg"))
wordcloud = WordCloud(mask=mask).generate(sample_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig("wordcloud.png")
fig = PIL.Image.open("wordcloud.png")
fig.show()"""

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