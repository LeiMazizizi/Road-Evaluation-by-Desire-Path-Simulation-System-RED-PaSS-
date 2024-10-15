import io

import dash
from dash import dcc, html
import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash.dependencies import Input,Output, State

import base64
import datetime
# import io
# import os
# import zipfile
# import shutil
# import tempfile
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features
import shapely

from shapely.affinity import affine_transform
from shapely.geometry import Polygon,Point,LineString,mapping
import json
from plotly import express as px
# from PIL import Image
# from io import BytesIO
from dash_canvas.utils import array_to_data_url,image_string_to_PILImage
# from PIL import TiffImagePlugin,ExifTags
from ABM_simulation_func import ABM_simulation,visulization_analysis_results

# dcc.U
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# title=dcc.Markdown(children="## **Desire Path Evaluation of Road Networks**")
title=dcc.Markdown('''
    # Road Evaluation by Desire Path Simulation System
    ####  by Lei Ma - University of Gävle
''',style={'font-family': 'Arial'})

### Old UI: ###
# app.layout=dbc.Container([title,
#     dcc.Store(id='boundingboxlist',data=[], storage_type='memory'),
#                           dcc.Store(id='rasterizedBuilding',data=[], storage_type='memory'),
#                           dcc.Store(id='rasterizedRoads',data=[], storage_type='memory'),
#                             dcc.Store(id='roadvectorlist',data=[], storage_type='memory'),
#
#     dcc.Store(id='ODdata-list',data=[], storage_type='memory'),
# dcc.Store(id='ODdata-flowQuantity',data=[], storage_type='memory'),
# # app.layout = html.Div([
#     dcc.Upload(id='upload-data', children=html.Div(['Building File...    Drag and Drop or ', html.A('Select Files')]),
#                 style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
#                        'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
#                 # Allow multiple files to be uploaded),
#                 multiple=False),
#     html.Div(id='output-data-upload'),
#     # dcc.Graph(id='map'),
#
#     dcc.Upload(id='upload-data-roads', children=html.Div(['Road File...    Drag and Drop or ', html.A('Select Files')]),
#                 style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
#                           'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
#                 # Allow multiple files to be uploaded),
#                 multiple=False),
#     html.Div(id='output-data-upload-roads'),
#
#
#
#     # dcc.Upload(id='upload-data-pop-tiff',
#     #          children=html.Div(['Population TIFF File...    Drag and Drop or ', html.A('Select Files')]),
#     #          style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
#     #                 'borderWidth': '1px',
#     #                 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center',
#     #                 'margin': '10px'},
#     #          # Allow multiple files to be uploaded),
#     #          multiple=False),
#     # html.Div(id='output-data-upload-tiff'),
#
#
# # app.layout = html.Div([
#     dcc.Upload(id='upload-ODs', children=html.Div(['Origin-Destination File...    Drag and Drop or ', html.A('Select Files')]),
#                 style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
#                        'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
#                 # Allow multiple files to be uploaded),
#                 multiple=False),
#     html.Div(id='output-data-ODs'),
#                           html.Hr(),
#
#                           dbc.Input(id='num_of_agents', type='number', value=100, placeholder='Number of agents'),
#                           dbc.Input(id='angle', type='number', value=100, placeholder='Angle of viewshed',),
#                           dbc.Input(id='depth', type='number', value=6, placeholder='Depth of viewshed',),
#                           dbc.Input(id='ticks', type='number', value=500, placeholder='Number of ticks',),
#                           dbc.Button(id='btn-run', n_clicks=0,children="Run",
#                                       style={'font-size': '16px', 'width': '100px', 'display': 'inline-block', 'margin-bottom': '10px', 'margin-right': '5px', 'height':'40px', 'verticalAlign': 'top'},),
#
#                           dbc.Button([dbc.Spinner(html.Div(id='run-result'),size='sm'), "  Results shown here"], id='running',
#                                    color="primary", disabled=True,),
#
#
#     # dcc.Graph(id='map-roads'),
#
#     # dcc.Graph(id='map-roads-2'),
#     # dcc.Graph(id='real-time-plot'),
#     # dcc.Graph(id='real-time-plot'),
#     # dcc.Interval(
#     #     id='interval-component',
#     #     interval=100,  # Update every 1000 milliseconds (1 second)
#     #     n_intervals=ticks
#     # )
# ])
### Old UI ###:
### New UI: ###
# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "30rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "31rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    # 'background-image': 'url("/assets/background.png")',
}
CONTENT_STYLE_button = {
    "margin-left": "31rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",

}

sidebar=dbc.Container([title,
html.Hr(),
html.P('Upload files:'),
    dcc.Store(id='boundingboxlist',data=[], storage_type='memory'),
                          dcc.Store(id='rasterizedBuilding',data=[], storage_type='memory'),
                          dcc.Store(id='rasterizedRoads',data=[], storage_type='memory'),
                            dcc.Store(id='roadvectorlist',data=[], storage_type='memory'),

    dcc.Store(id='ODdata-list',data=[], storage_type='memory'),
dcc.Store(id='ODdata-flowQuantity',data=[], storage_type='memory'),
# app.layout = html.Div([
    dcc.Upload(id='upload-data', children=html.Div(['Affordance File...    Drag and Drop or ', html.A('Select Files')]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                       'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                # Allow multiple files to be uploaded),
                multiple=False),
    # html.Div(id='output-data-upload'),
    # dcc.Graph(id='map'),

    dcc.Upload(id='upload-data-roads', children=html.Div(['Road File...    Drag and Drop or ', html.A('Select Files')]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                          'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                # Allow multiple files to be uploaded),
                multiple=False),




    # dcc.Upload(id='upload-data-pop-tiff',
    #          children=html.Div(['Population TIFF File...    Drag and Drop or ', html.A('Select Files')]),
    #          style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
    #                 'borderWidth': '1px',
    #                 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center',
    #                 'margin': '10px'},
    #          # Allow multiple files to be uploaded),
    #          multiple=False),
    # html.Div(id='output-data-upload-tiff'),


# app.layout = html.Div([
    dcc.Upload(id='upload-ODs', children=html.Div(['Origin-Destination(OD) File...    Drag and Drop or ', html.A('Select Files')]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                       'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                # Allow multiple files to be uploaded),
                multiple=False),
    # html.Div(id='output-data-ODs'),
                          html.Hr(),
                        html.P('Number of agents:'),
                          dbc.Input(id='num_of_agents', type='number', value=100, placeholder='Number of agents'),
                            html.P('Angle of viewshed:'),
                          dbc.Input(id='angle', type='number', value=100, placeholder='Angle of viewshed',),
                            html.P('Depth of viewshed:'),
                          dbc.Input(id='depth', type='number', value=6, placeholder='Depth of viewshed',),
                            html.P('Number of ticks:'),
                          dbc.Input(id='ticks', type='number', value=500, placeholder='Number of ticks',),
                          dbc.Button(id='btn-run', n_clicks=0,children="Run",
                                      style={'font-size': '16px', 'width': '100px', 'display': 'inline-block', 'margin-bottom': '10px', 'margin-right': '5px', 'height':'40px', 'verticalAlign': 'top'},),




    # dcc.Graph(id='map-roads'),

    # dcc.Graph(id='map-roads-2'),
    # dcc.Graph(id='real-time-plot'),
    # dcc.Graph(id='real-time-plot'),
    # dcc.Interval(
    #     id='interval-component',
    #     interval=100,  # Update every 1000 milliseconds (1 second)
    #     n_intervals=ticks
    # )
],style=SIDEBAR_STYLE)
content = dbc.Container(children=[html.Div([html.Img(id='bg-image',src='assets/background.png',style={'width':'20%','height':'20%','z-index':'-1'},)],style={  'top':'0px','left':'500px'}),
                        html.Div(id='output-data-upload'),
                        html.Div(id='output-data-upload-roads'),
                        html.Div(id='output-data-ODs'),
                        html.Div(id="page-content"),
                        dbc.Button(
                          [dbc.Spinner(html.Div(id='run-result'), size='sm'), "Report shown here"],
                          id='running',
                          color="secondary", disabled=True, className="me-1",),
                                  ],style=CONTENT_STYLE
                        )

style_layout={
    'background-image': 'url("/assets/background.png")',
    # 'background-size': 'cover',
    # 'background-repeat': 'no-repeat',
    # 'background-position': 'center',
    # 'height': '50vh'
}
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])
### New UI ###




# fig = go.Figure(data=go.Image(z=np.stack((np.flip(rasterized_world, 0),) * 3, -1)))
# world_visualization(position_data,rasterized_world,rasterized_world_shape,fig)
# fig.show()

# @app.callback(
#     # dash.dependencies.Output('real-time-plot', 'figure'),
#     # [dash.dependencies.Input('interval-component', 'n_intervals')]
# )

def parse_contents_building(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    bb=[]
    if contents is None:
        return 'No file uploaded.',bb
    # # Check if the uploaded file is a ZIP archive
    # if not filename.endswith('.zip'):
    #     return 'Please upload a .zip file containing the Shapefile.'

    # # Create a temporary directory to store the uploaded files
    # temp_dir = 'temp_upload'
    # os.makedirs(temp_dir, exist_ok=True)
    # print(os.path.join(temp_dir, filename))
    # # if filename.endswith('.shp'):
    try:

        #
        # # Save the uploaded ZIP file to the temporary directory
        # with open(os.path.join(temp_dir, filename), 'wb') as file:
        #     file.write(contents)
        # # Extract the contents of the ZIP file
        # with zipfile.ZipFile(os.path.join(temp_dir, filename), 'r') as zip_ref:
        #     zip_ref.extractall(temp_dir)
        # print(os.path.join(temp_dir, filename))
        #
        # # Search for the .shp file within the extracted files
        # shp_file = None
        # for root, dirs, files in os.walk(temp_dir):
        #     for file in files:
        #         if file.endswith('.shp'):
        #             shp_file = os.path.join(root, file)
        #             break
        # if shp_file:
        #     # Read the Shapefile using geopandas
        #     print(shp_file)
        #     vector = gpd.read_file(shp_file)
        #     return f'Shapefile uploaded: {filename}'
        # else:
        #     return 'No .shp file found in the uploaded ZIP.'

        # # Create a temporary directory
        # with tempfile.TemporaryDirectory() as temp_dir:
        #     # Generate a temporary file path
        #     temp_file_path = os.path.join(temp_dir, filename)
        #
        #     # Write the uploaded file's contents to the temporary file
        #     with open(temp_file_path, 'wb') as temp_file:
        #         temp_file.write(contents.encode('utf-8'))
        # print(temp_file_path)


        # vector=pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        dic=json.loads(decoded.decode('utf-8'))
        crs=dic['crs']['properties']['name']
        # Create an empty list to store geometries
        geometries = []
        geo={'type':'Polygon'}


        for _ in dic['features']:
            # dict['geometry']=shapely.geometry.shape(_['geometry'])
            geo['coordinates']=[_['geometry']['coordinates'][0]]
            geometries.append(shapely.geometry.shape(geo))

            geo={'type':'Polygon'}

        vector = gpd.GeoDataFrame(crs=crs, geometry=geometries)


        # vector = gpd.read_file(content_string)
        # features = json.load(vector)["features"]


        # if len(gloabl_boundingBoxList) == 0:

        bb1 = vector.bounds
        minx = bb1.minx.min()
        maxx = bb1.maxx.max()
        miny = bb1.miny.min()
        maxy = bb1.maxy.max()
        bb = vector['geometry'].total_bounds
        nodataFill = 1
        valueFill = 0
        # else:
        #     minx = gloabl_boundingBoxList[0]
        #     maxx = gloabl_boundingBoxList[2]
        #     miny = gloabl_boundingBoxList[1]
        #     maxy = gloabl_boundingBoxList[3]
        #     bb = np.array([minx, miny, maxx, maxy])

        gloabl_boundingBoxList=bb
        # print(minx,maxx,miny,maxy)
        # get extent values to set size of output raster.
        # x_min, x_max, y_min, y_max = vector.GetExtent()
        xr = maxx - minx
        yr = maxy - miny
        ratioofxy = xr / yr

        # Get list of geometries for all features in vector file
        geom = [shapes for shapes in vector.geometry]

        # # Open example raster
        # raster = rasterio.open(r"F:\task_pool\scalesHumanMobility\angle_city\data\London\textdata\building2.tif")
        # rt=raster.transform
        shape = 100, int(100 / ratioofxy)

        transform = rasterio.transform.from_bounds(*bb, *shape)

        # Rasterize vector using the shape and coordinate system of the raster
        rasterized = features.rasterize(geom,
                                        # out_shape = raster.shape,
                                        out_shape=shape,
                                        fill=nodataFill,
                                        out=None,
                                        transform=transform,
                                        all_touched=True,
                                        default_value=valueFill,
                                        dtype=None)

        image_data=np.stack((np.flip(rasterized, 0),) * 3, -1).tolist()
        # image_data = np.stack((rasterized,) * 3, -1).tolist()
        # img = image_data.astronaut()  # numpy array
        # pil_img = Image.fromarray(image_data)  # PIL image object
        # prefix = "data:image/png;base64,"
        # with BytesIO() as stream:
        #     pil_img.save(stream, format="png")
        #     base64_string = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")


        # fig = go.Figure(go.Image(z=image_data))
        rasterized_flipped=np.flip(rasterized, 0).tolist()

        fig=go.Figure(data=go.Heatmap(z=rasterized_flipped))
        fig.update_layout(width=600,height=600)
        return html.Div([
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),
            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            # html.Img(src=),
            dcc.Graph(figure=fig),
            # dcc.Store(id='boundingboxlist', data=rasterized,storage_type='local'),

            html.Hr(),
            # html.Div('Raw Content'),
            # html.Pre(contents[0:200] + '...', style={
            #     'whiteSpace': 'pre-wrap',
            #     'wordBreak': 'break-all'
            # })
        ]),bb,rasterized
    except Exception as e:
        return f'Error reading Shapefile: {e}',bb,[]
    # finally:
    #     # Clean up: remove temporary directory and files
    #     shutil.rmtree(temp_dir, ignore_errors=True)
    # else:
    #     return 'Please upload a Shapefile.'

def parse_contents_roads(contents, filename, date,gloabl_boundingBoxList):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if contents is None:
        return 'No file uploaded.'

    # decoded= base64.b64decode(content_string)
    print(gloabl_boundingBoxList)
    try:
        dic=json.loads(decoded.decode('utf-8'))
        crs=dic['crs']['properties']['name']
        # Create an empty list to store geometries
        geometries = []

        for _ in dic['features']:
            geo = {}
            geo['coordinates'] = _['geometry']['coordinates']
            try:

                # dict['geometry']=shapely.geometry.shape(_['geometry'])

                # flat_list = [item for sublist in _['geometry']['coordinates'][0] for item in sublist]

                geo['type']= 'LineString'
                geometries.append(shapely.geometry.shape(geo))
            except Exception as e:
                geo['type'] = 'MultiLineString'
                geometries.append(shapely.geometry.shape(geo))
                # geo={'type':'LineString'}

        vector = gpd.GeoDataFrame(crs=crs, geometry=geometries)


        # vector = gpd.read_file(contents)
# try:
#     if 'shp' in filename:
#         # Read in vector
#         global gloabl_boundingBoxList
#         if len(gloabl_boundingBoxList) != 0:
        minx = gloabl_boundingBoxList[0]
        maxx = gloabl_boundingBoxList[2]
        miny = gloabl_boundingBoxList[1]
        maxy = gloabl_boundingBoxList[3]
        bb = np.array([minx, miny, maxx, maxy])

        # gloabl_boundingBoxList=bb
        # print(minx,maxx,miny,maxy)
        # get extent values to set size of output raster.
        # x_min, x_max, y_min, y_max = vector.GetExtent()
        xr = maxx - minx
        yr = maxy - miny
        ratioofxy = xr / yr

        # Get list of geometries for all features in vector file
        geom = [shapes for shapes in vector.geometry]

        # # Open example raster
        # raster = rasterio.open(r"F:\task_pool\scalesHumanMobility\angle_city\data\London\textdata\building2.tif")
        # rt=raster.transform
        shape = 100, int(100 / ratioofxy)

        transform = rasterio.transform.from_bounds(*bb, *shape)
        nodataFill = 0
        valueFill = 255
        # Rasterize vector using the shape and coordinate system of the raster
        rasterized = features.rasterize(geom,
                                        # out_shape = raster.shape,
                                        out_shape=shape,
                                        fill=nodataFill,
                                        out=None,
                                        transform=transform,
                                        all_touched=True,
                                        default_value=valueFill,
                                        dtype=None)




        scale_x = (100 - 0) / (maxx - minx)
        scale_y = (100 - 0) / (maxy - miny)

        matrix = [scale_x, 0, 0, scale_y, 0 - scale_x * minx, 0 - scale_y * miny]

        new_geoms = []
        for _ in geom:
            new_g = affine_transform(_, matrix)
            new_geoms.append(new_g)

        gdfh = gpd.GeoDataFrame(geometry=new_geoms)
        gdfh = gdfh.explode()
        geom = gdfh.geometry

        # fig = go.Figure(go.Image(z=np.stack((rasterized,) * 3, -1)))
        # fig = go.Figure(go.Image(z = np.stack((np.flip(rasterized, 0),) * 3, -1)))

        flipped_rasterized = np.flip(rasterized, 0)
        fig = go.Figure( data=go.Heatmap(z=flipped_rasterized))
        for index, row in gdfh.iterrows():
            _ = row['geometry']

            x = _.coords.xy[0].tolist()
            y = _.coords.xy[1].tolist()
            # x, y = _.coords.xy
            fig.add_trace(
                go.Scatter(x=x, y=y, line=dict(width=1), showlegend=False)
                # px.line(x=x, y=y, render_mode="svg")
            )
        fig.update_layout(width=600, height=600)

        return html.Div([
            html.H5(filename),
            # html.H6(datetime.datetime.fromtimestamp(date)),
            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            # html.Img(src=np.flip(rasterized, 0)),
            dcc.Graph(figure=fig),
            html.Hr(),
            # html.Div('Raw Content'),
            # html.Pre(contents[0:200] + '...', style={
            #     'whiteSpace': 'pre-wrap',
            #     'wordBreak': 'break-all'
            # })
        ]),rasterized,geom

    except Exception as e:
        return f'Error reading Shapefile: {e}',[],[]


def parse_contents_tiff(list_of_contents, list_of_names, list_of_dates):
    # Using base64 encoding and decoding
    content_type, content_string = list_of_contents.split(',')



    img=image_string_to_PILImage(list_of_contents)
    # = img2 = Image.open(BytesIO(base64.b64decode(content_string)))
    pix=np.array(img)
    img_content=array_to_data_url(pix)
    fig = go.Figure(go.Image(z=np.stack((pix,)*3,-1)))
    # mm=img.getprojection()
    # mm=img.getbbox()

    # exif = {ExifTags.TAGS[k]: v for k, v in img.getexif().items() if k in ExifTags.TAGS}



    # gt = img.transform
    # pixelSizeX = gt[0]
    # pixelSizeY = -gt[4]
    # cellarea=pixelSizeY*pixelSizeX


    return html.Div([
        dcc.Graph(figure=fig),
        # html.Img(src=image_path),  # passing the direct file path
        # html.Img(src=app.get_asset_url('my-image.png')),  # usign get_asset_url function
        # html.Img(src=dash.get_asset_url('my-image.png'))    Or with newer Dash v2.2.0
        # html.Img(src=content_string),  # using the pillow image variable
        # html.Img(src=b64_image(image_path)),  # using base64 to encode and decode the image file

    ])

def parse_contents_ODs(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    bb=[]
    try:
        if contents is None:
            return 'No file uploaded.',bb
        datas=[[],[]]

        df= pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        for index,row in df.iterrows():
            datas[0].append([(row['OriginX'],row['OriginY']),(row['DestinationX'],row['DestinationY'])])
            datas[1].append(row['FlowQuantity'])
        bb=datas
        fig=px.histogram(df,x='FlowQuantity',nbins=30)
        return html.Div([
            html.H5(filename),
            # html.H6(datetime.datetime.fromtimestamp(date)),
            dcc.Graph(figure=fig),
            html.Hr(),



            # })
        ]),bb
    except Exception as e:
        return html.Div(f'Error reading csv file: {e}'),bb

@app.callback([Output('output-data-upload', 'children'),
              Output('boundingboxlist', 'data'),
               Output('rasterizedBuilding', 'data')],
                        [Input('upload-data', 'contents'),
                        State('upload-data', 'filename'),
                        State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        # print(list_of_contents)
        # print(list_of_names)
        # print(list_of_dates)
        # children= [
        #     parse_contents_building(c, n, d) for c, n, d in
        # zip(list_of_contents, list_of_names, list_of_dates)]
        result,bb, building_rasterized=parse_contents_building(list_of_contents, list_of_names, list_of_dates)
        return result,bb,building_rasterized
    else:
        return 'No affordance file uploaded.',[],[]

@app.callback([Output('output-data-upload-roads', 'children'),
              Output('roadvectorlist', 'data'),
               Output('rasterizedRoads','data')],

              [Input('upload-data-roads', 'contents'),
                        State('upload-data-roads', 'filename'),
                        State('upload-data-roads', 'last_modified'),
              Input('boundingboxlist', 'data')])
def update_output2(list_of_contents2, list_of_names2, list_of_dates2,bb):
    if list_of_contents2 is not None:
        # print(list_of_contents)
        # print(list_of_names)
        # print(list_of_dates)
        # children= [
        #     parse_contents_building(c, n, d) for c, n, d in
        # zip(list_of_contents, list_of_names, list_of_dates)]
        result, rasterized, road_list=parse_contents_roads(list_of_contents2, list_of_names2, list_of_dates2,bb)

        geos= []
        for _ in road_list:
            geos.append(mapping(_))

        return result,geos,rasterized
    else:
        return 'No road file uploaded.',[],[]

# @app.callback(Output('output-data-upload-tiff', 'children'),
#                         Input('upload-data-pop-tiff', 'contents'),
#                         State('upload-data-pop-tiff', 'filename'),
#                         State('upload-data-pop-tiff', 'last_modified'))
# def update_output3(list_of_contents3, list_of_names3, list_of_dates3):
#     if list_of_contents3 is not None:
#         # print(list_of_contents)
#         # print(list_of_names)
#         # print(list_of_dates)
#         # children= [
#         #     parse_contents_building(c, n, d) for c, n, d in
#         # zip(list_of_contents, list_of_names, list_of_dates)]
#         result=parse_contents_tiff(list_of_contents3, list_of_names3, list_of_dates3)
#         return result
#     return 'No population jpeg uploaded.'

@app.callback([Output('output-data-ODs', 'children'),
              Output('ODdata-list', 'data'),
               Output('ODdata-flowQuantity','data')],
                        [Input('upload-ODs', 'contents'),
                        State('upload-data', 'filename'),
                        State('upload-data', 'last_modified')])
def update_output4(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        # print(list_of_contents)
        # print(list_of_names)
        # print(list_of_dates)
        # children= [
        #     parse_contents_building(c, n, d) for c, n, d in
        # zip(list_of_contents, list_of_names, list_of_dates)]
        result,bb=parse_contents_ODs(list_of_contents, list_of_names, list_of_dates)
        return result,bb[0],bb[1]
    else:
        return 'No OD .csv file uploaded.',[],[]

@app.callback(Output('run-result', 'children'),
[Input('btn-run', 'n_clicks'),
 Input('boundingboxlist', 'data'),
 Input('rasterizedBuilding', 'data'),
 Input('rasterizedRoads', 'data'),
 Input('roadvectorlist', 'data'),
 Input('ODdata-list', 'data'),
    Input('ODdata-flowQuantity', 'data'),
 Input('num_of_agents', 'value'),
 Input('angle', 'value'),
 Input('depth', 'value'),
 Input('ticks', 'value')])
def update_output5(n_clicks,bb,building_rasterized,road_rasterized,road_list,OD_list,OD_flowQuantity_list,
                   num_of_agents,angle,depth,tick):
    if n_clicks is not None:
        if n_clicks == 1:

            road_vector_shapely_lines= []
            for _ in road_list:
                road_vector_shapely_lines.append(shapely.geometry.shape(_))

            RASTERIZED_WORLD_breaks,RASTERIZED_WORLD=ABM_simulation(building_rasterized, road_rasterized,np.array(building_rasterized).shape, road_vector_shapely_lines,
                       OD_list, OD_flowQuantity_list, num_of_agents,
                       angle, depth, 1,tick)

            fig_path_of_breaks, fig_spatial_overlay_with_paths, roads_on_paths_ratios=visulization_analysis_results(RASTERIZED_WORLD_breaks,
                                                                                                                    RASTERIZED_ROADS=np.array(road_rasterized),
                                                                                                                    GEOM_ROADS=road_vector_shapely_lines,
                                                                                                                    display_colors=3)

            return html.Div([
            html.H5("Analysis Report:"),
            # html.H6(datetime.datetime.fromtimestamp(date)),
            dcc.Graph(figure=fig_path_of_breaks),
                html.Hr(),
                html.Plaintext("% of paths on the roads at four levels separately: {}".format(roads_on_paths_ratios)),
                html.Hr(),
                dcc.Graph(figure=fig_spatial_overlay_with_paths),
            html.Hr(),



            # })
        ])
        else:
            return []
    else:
        return 'Press the Run button!'


def update_real_time_plot(n_intervals):
    # Generate some random data (replace this with your own data source)
    # x_data = [1, 2, 3, 4, 5]
    # y_data = [random.randint(0, 100) for _ in x_data]
    # global POSITION_DATA
    # fig = go.Figure(data=go.Image(z=np.stack((np.flip(rasterized_world, 0),) * 3, -1)))
    pass
    # world_visualization(rasterized_world, rasterized_world_shape, fig)

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines+markers', name='Data'))
    # fig.update_layout()
if __name__ == '__main__':
    app.run_server(debug=True)