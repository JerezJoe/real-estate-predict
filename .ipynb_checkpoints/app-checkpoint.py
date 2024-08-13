from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from rtree import index
import requests as req
import pickle
import joblib
import numpy as np

app = Flask(__name__) 
CORS(app, resources={r"/*": {"origins": "*"}})

uni_df = pd.read_csv("/external_data/大學基本資料.csv")
uni_df.drop(uni_df[(uni_df['lng'] > 126) | (uni_df['lng'] < 119)].index, inplace=True)
uni_df.drop(uni_df[(uni_df["lat"] > 26) | (uni_df["lat"] < 20)].index, inplace=True)

bus_df = pd.read_csv("/external_data/公車站點資料.csv")
bus_df.drop(bus_df[(bus_df['lng'] > 126) | (bus_df['lng'] < 119)].index, inplace=True)
bus_df.drop(bus_df[(bus_df["lat"] > 26) | (bus_df["lat"] < 20)].index, inplace=True)

train_df = pd.read_csv("/external_data/火車站點資料.csv")
train_df.drop(train_df[(train_df['lng'] > 126) | (train_df['lng'] < 119)].index, inplace=True)
train_df.drop(train_df[(train_df["lat"] > 26) | (train_df["lat"] < 20)].index, inplace=True)

bank_df = pd.read_csv("/external_data/金融機構基本資料.csv")
bank_df.drop(bank_df[(bank_df['lng'] > 126) | (bank_df['lng'] < 119)].index, inplace=True)
bank_df.drop(bank_df[(bank_df["lat"] > 26) | (bank_df["lat"] < 20)].index, inplace=True)

conv_df = pd.read_csv("/external_data/便利商店.csv")
conv_df.drop(conv_df[(conv_df['lng'] > 126) | (conv_df['lng'] < 119)].index, inplace=True)
conv_df.drop(conv_df[(conv_df["lat"] > 26) | (conv_df["lat"] < 20)].index, inplace=True)
conv_df.dropna(axis = 0,how = 'any' ,inplace=True)

highs_df = pd.read_csv("/external_data/高中基本資料.csv")
highs_df.drop(highs_df[(highs_df['lng'] > 126) | (highs_df['lng'] < 119)].index, inplace=True)
highs_df.drop(highs_df[(highs_df["lat"] > 26) | (highs_df["lat"] < 20)].index, inplace=True)

elem_df = pd.read_csv("/external_data/國小基本資料.csv")
elem_df.drop(elem_df[(elem_df['lng'] > 126) | (elem_df['lng'] < 119)].index, inplace=True)
elem_df.drop(elem_df[(elem_df["lat"] > 26) | (elem_df["lat"] < 20)].index, inplace=True)

juni_df = pd.read_csv("/external_data/國中基本資料.csv")
juni_df.drop(juni_df[(juni_df['lng'] > 126) | (juni_df['lng'] < 119)].index, inplace=True)
juni_df.drop(juni_df[(juni_df["lat"] > 26) | (juni_df["lat"] < 20)].index, inplace=True)

hospi_df = pd.read_csv("/external_data/醫療機構基本資料.csv")
hospi_df.drop(hospi_df[(hospi_df['lng'] > 126) | (hospi_df['lng'] < 119)].index, inplace=True)
hospi_df.drop(hospi_df[(hospi_df["lat"] > 26) | (hospi_df["lat"] < 20)].index, inplace=True)

road_df = pd.read_csv("/external_data/快速公路交流道里程及通往地名_11211.csv")
road_df['lat'] = road_df['WGS84-Y']
road_df['lng'] = road_df['WGS84-X']

mrt_df = pd.read_csv("/external_data/northern-taiwan.csv")
mrt_df['lng'] = mrt_df['lon']
uni_idx_rtree = index.Index()
train_idx_rtree = index.Index()
highs_idx_rtree = index.Index()
elem_idx_rtree = index.Index()
juni_idx_rtree = index.Index()
hospi_idx_rtree = index.Index()
conv_idx_rtree = index.Index()
bank_idx_rtree = index.Index()
bus_idx_rtree = index.Index()
road_idx_rtree = index.Index()
mrt_idx_rtree = index.Index()
def input_idx_tree(df, tree):
    for idx, row in df.iterrows():
        lat, lng = row['lat'], row['lng']
        tree.insert(idx, (lng, lat, lng, lat)) 
    return 0
        
input_idx_tree(uni_df, uni_idx_rtree)
input_idx_tree(train_df, train_idx_rtree)
input_idx_tree(highs_df, highs_idx_rtree)
input_idx_tree(elem_df, elem_idx_rtree)
input_idx_tree(juni_df, juni_idx_rtree)
input_idx_tree(hospi_df, hospi_idx_rtree)
input_idx_tree(conv_df, conv_idx_rtree)
input_idx_tree(bank_df, bank_idx_rtree)
input_idx_tree(bus_df, bus_idx_rtree)
input_idx_tree(road_df, road_idx_rtree)
input_idx_tree(mrt_df, mrt_idx_rtree)

def counting(df):
    global uni_idx_rtree
    global train_idx_rtree 
    global highs_idx_rtree 
    global elem_idx_rtree
    global juni_idx_rtree 
    global hospi_idx_rtree 
    global conv_idx_rtree 
    global bank_idx_rtree 
    global bus_idx_rtree 
    global road_idx_rtree 
    global mrt_idx_rtree
    radius = 2 / 111.32  # 1公里的經度和緯度換算關係（約每緯度111.32公里）

# 執行範圍查詢
    uni_count_list = []
    train_count_list = []
    highs_count_list = []
    elem_count_list = []
    juni_count_list = []
    hospi_count_list = []
    conv_count_list = []
    bank_count_list = []
    bus_count_list = []
    road_count_list = []
    mrt_count_list = []
    for h_idx, h_row in df.iterrows():
        lat, lng = h_row['lat'], h_row['lng']
        count = 0
        for intersect_id in uni_idx_rtree.intersection((lng - radius, lat - radius, lng + radius, lat + radius)):
            count += 1
        uni_count_list.append(count)
        count = 0
        for intersect_id in train_idx_rtree.intersection((lng - radius, lat - radius, lng + radius, lat + radius)):
            count += 1
        train_count_list.append(count)
        count = 0
        for intersect_id in highs_idx_rtree.intersection((lng - radius, lat - radius, lng + radius, lat + radius)):
            count += 1
        highs_count_list.append(count)
        count = 0
        for intersect_id in elem_idx_rtree.intersection((lng - radius, lat - radius, lng + radius, lat + radius)):
            count += 1
        elem_count_list.append(count)
        count = 0
        for intersect_id in juni_idx_rtree.intersection((lng - radius, lat - radius, lng + radius, lat + radius)):
            count += 1
        juni_count_list.append(count)
        count = 0
        for intersect_id in bank_idx_rtree.intersection((lng - radius, lat - radius, lng + radius, lat + radius)):
            count += 1
        bank_count_list.append(count)
        count = 0
        for intersect_id in conv_idx_rtree.intersection((lng - radius, lat - radius, lng + radius, lat + radius)):
            count += 1
        conv_count_list.append(count)
        count = 0
        for intersect_id in bus_idx_rtree.intersection((lng - radius, lat - radius, lng + radius, lat + radius)):
            count += 1
        bus_count_list.append(count)
        count = 0
        for intersect_id in road_idx_rtree.intersection((lng - radius, lat - radius, lng + radius, lat + radius)):
            count += 1
        road_count_list.append(count)
        count = 0
        for intersect_id in mrt_idx_rtree.intersection((lng - radius, lat - radius, lng + radius, lat + radius)):
            count += 1
        mrt_count_list.append(count)
        count = 0
        for intersect_id in hospi_idx_rtree.intersection((lng - radius, lat - radius, lng + radius, lat + radius)):
            count += 1
        hospi_count_list.append(count)

    df['uni_count'] = uni_count_list
    df['train_count'] = train_count_list
    df['highs_count'] = highs_count_list
    df['elem_count'] = elem_count_list
    df['juni_count'] = juni_count_list
    df['hospi_count'] = hospi_count_list
    df['conv_count'] = conv_count_list
    df['bank_count'] = bank_count_list
    df['bus_count'] = bus_count_list
    df['road_count'] = road_count_list
    df['mrt_count'] = mrt_count_list
    f_lst = df['移轉層次_n']
    t_lst = df['總樓層數']
    f_percent = [round(f/t, 2) if t != 0 else 0 for f, t in zip(f_lst, t_lst)]
    df['樓層比'] = f_percent
    return df
def get_scaler(df):
    df_val = df.copy()
    scaler_1 = joblib.load('scaler_area_only.joblib')
    scaler_4 = joblib.load('scaler_areaCar_only.joblib')
    area = np.array(df_val['建物面積_n']).reshape(-1,1)
    area_c = np.array(df_val['車位移轉總面積平方公尺']).reshape(-1,1)
    X_1 = scaler_1.transform(area)
    X_4 = scaler_4.transform(area_c)
    df['建物面積_n'] = X_1
    df['車位移轉總面積平方公尺'] = X_4
    return df
def get_map(df):
    with open('map_dict_36.pkl', 'rb') as file:
        dicts = pickle.load(file)
    df['縣市_mapped'] = df['縣市'].map(dicts[0])
    df['建物型態_mapped'] = df['建物型態'].map(dicts[1])
    df['鄉鎮市區_mapped'] = df['鄉鎮市區'].map(dicts[4])
    df['主要建材_mapped'] = df['主要建材'].map(dicts[5])
    return df
def get_att(address):
    url = 'https://maps.googleapis.com/maps/api/geocode/json?'

    api_key = 'AIzaSyCLkuN16-YxrOKu6eBZBcOYkc0922GTaFI'

    response = req.get(url + 'address=' + address + '&key=' + api_key)

    data = response.json()
    if data['status'] == 'OK':
        lat = data['results'][0]['geometry']['location']['lat']
        lng = data['results'][0]['geometry']['location']['lng']
        return lat,lng
    else:
        return "",""
def trans_add(df):
    lat_l = []
    lng_l = []
    for address in df['地址']:
        lat, lng = get_att(address)
        lat_l.append(lat)
        lng_l.append(lng)
    df['lat'] = lat_l
    df['lng'] = lng_l
    return df

def predict(df):
    features = ['總樓層數',
             '建物現況格局-房',
             '建物現況格局-廳',
             '建物現況格局-衛',
             '車位移轉總面積平方公尺',
             '建物面積_n',
             '移轉層次_n',
             '縣市_mapped',
             '建物型態_mapped',
             '鄉鎮市區_mapped',
             '主要建材_mapped',
             'uni_count',
             'train_count',
             'highs_count',
             'elem_count',
             'juni_count',
             'hospi_count',
             'conv_count',
             'bank_count',
             'bus_count',
             'road_count',
             'mrt_count',
             '樓層比'
            ]
    X = df[features]
    xgb_model = joblib.load('xgb_27.joblib')
    price_scaler = joblib.load('scaler_price_only.joblib')
    y_pred = xgb_model.predict(X)
    result = y_pred[0]
    result = np.array(result).reshape(-1, 1)
    result = price_scaler.inverse_transform(result)
    df['預測結果'] = result
    return df 
@app.route('/receive-json', methods=['POST'])
def receive_json():
    df = pd.read_csv('api_test.csv')
    df = get_map(df)
    df = get_scaler(df)
    df = trans_add(df)
    df = counting(df)
    df = predict(df)
    result = df['預測結果'].tolist()
    if request.method == 'POST':
        received_data = request.json
       #  df = pd.DataFrame(received_data)
       #  df.columns = ['總樓層數', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛', '車位移轉總面積平方公尺', '建物面積_n',
       # '移轉層次_n','縣市','鄉鎮市區','地址','建物型態','主要建材']
       #  df = get_map(df)
       #  df = get_scaler(df)
       #  df = trans_add(df)
       #  df = counting(df)
       #  df = predict(df)
        return jsonify({'message': '傳送成功','prediction': result})

if __name__ == '__main__':
    app.run(debug=True)

